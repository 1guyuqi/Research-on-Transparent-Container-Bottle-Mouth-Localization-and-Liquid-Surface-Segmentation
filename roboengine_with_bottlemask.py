# -*- coding: utf-8 -*-
"""
dealbottle 流程：
  - 机械臂：由 RoboEngine 自动分割（RoboEngineRobotSegmentation）。
  - 瓶子：用你提供的 container_mask.png 替代 RoboEngine 的瓶子分割，不调用物体分割模型。
  - 合并两者 mask 后做纹理背景增强。
用法示例（在项目根目录下）：
  python dealbottle_with_mask.py --image 原图.jpg --mask container_mask.png
  （不加 --no-robot 时默认会调用 RoboEngine 分割机械臂）
"""
import os
import random
import argparse
import sys
import traceback
from pathlib import Path

# 保证从项目根目录可导入 robo_engine（任意目录执行脚本时）
_script_dir = Path(__file__).resolve().parent
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))

import cv2
import numpy as np

# 可选：若需要机器人 mask，取消下面两行注释并安装 robo_engine 依赖
# from robo_engine.infer_engine import RoboEngineRobotSegmentation

# ---------- 默认路径（可通过命令行覆盖） ----------
DEFAULT_IMAGE_PATH = "bottle_image1.jpg"
DEFAULT_MASK_PATH = r"C:\Users\26370\Desktop\container_mask.png"
DEFAULT_TEXTURE_ROOT = None  # 默认使用脚本所在目录下的 textures/mil_data
DEFAULT_NUM_AUG = 3


def load_mask(mask_path: str, expected_hw=None):
    """加载二值 mask，保证为 (H,W) float32，前景>0 为 1.0"""
    raw = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if raw is None:
        raise FileNotFoundError(f"Mask not found: {os.path.abspath(mask_path)}")
    mask = (raw > 0).astype(np.float32)
    if expected_hw is not None:
        h, w = expected_hw[:2]
        if mask.shape[0] != h or mask.shape[1] != w:
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    return mask


def main():
    parser = argparse.ArgumentParser(description="dealbottle: 用已有瓶子 mask 做纹理背景增强")
    parser.add_argument("--image", type=str, default=DEFAULT_IMAGE_PATH, help="原图路径")
    parser.add_argument("--mask", type=str, default=DEFAULT_MASK_PATH, help="瓶子/容器二值 mask 路径（如 container_mask.png）")
    parser.add_argument("--texture-dir", type=str, default=None, help="纹理图目录（默认: 脚本目录/textures/mil_data）")
    parser.add_argument("--num-aug", type=int, default=DEFAULT_NUM_AUG, help="生成增强图数量")
    parser.add_argument("--out-mask", type=str, default="mask.png", help="保存合并 mask 的路径")
    parser.add_argument("--no-robot", action="store_true", help="不分割机械臂，仅用瓶子 mask；默认会调用 RoboEngine 自动分割机械臂")
    parser.add_argument("--debug", action="store_true", help="额外保存 robo_mask.png / bottle_mask.png 便于核对合并结果")
    args = parser.parse_args()

    # 1. 读取原图（使用绝对路径，避免后续切换工作目录后找不到文件）
    image_path = os.path.abspath(os.path.expanduser(args.image))
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Image not found: {os.path.abspath(image_path)}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    h, w = img_rgb.shape[0], img_rgb.shape[1]

    # 2. 瓶子：用你提供的 mask 替代 RoboEngine 的瓶子分割
    mask_path_abs = os.path.abspath(os.path.expanduser(args.mask))
    bottle_mask = load_mask(mask_path_abs, img_rgb.shape)
    bottle_mask = (bottle_mask > 0).astype(np.float32)
    assert bottle_mask.shape[0] == h and bottle_mask.shape[1] == w, "瓶子 mask 尺寸应与原图一致"

    # 3. 机械臂：RoboEngine 自动分割（--no-robot 时跳过）
    if args.no_robot:
        robo_mask = np.zeros((h, w), dtype=np.float32)
        print("[dealbottle] 未使用机械臂分割（--no-robot），仅用瓶子 mask")
    else:
        script_dir = Path(__file__).resolve().parent
        # RoboEngine 的 hydra 使用 config_path="."，且 compose 查找 robo_sam/...，故需在 robo_engine 目录下
        robo_engine_dir = script_dir / "robo_engine"
        cwd_before = os.getcwd()
        try:
            os.chdir(robo_engine_dir)
            from robo_engine.infer_engine import RoboEngineRobotSegmentation
            engine_robo_seg = RoboEngineRobotSegmentation()
            robo_mask = engine_robo_seg.gen_image(img_rgb, prompt="robot")
            robo_mask = (robo_mask > 0).astype(np.float32)
            # 保证与原图同尺寸（模型可能返回其他分辨率）
            if robo_mask.shape[0] != h or robo_mask.shape[1] != w:
                robo_mask = cv2.resize(robo_mask, (w, h), interpolation=cv2.INTER_NEAREST)
                robo_mask = (robo_mask > 0.5).astype(np.float32)
            print(f"[dealbottle] 机械臂 mask 已生成，前景像素数: {int(robo_mask.sum())}")
        except Exception as e:
            print("[dealbottle] RoboEngine 机械臂分割失败，将仅使用瓶子 mask。完整错误如下：", file=sys.stderr)
            traceback.print_exc()
            print("\n若不需要机械臂，可加 --no-robot 跳过。若需要机械臂，请确保：", file=sys.stderr)
            print("  1) 在项目根目录执行：cd <roboengine-main 所在目录> 再运行本脚本；", file=sys.stderr)
            print("  2) 已安装 robo_engine 依赖且能访问 HuggingFace 模型（michaelyuanqwq/roboengine-sam 等）；", file=sys.stderr)
            print("  3) 有 GPU 或已设置 CPU 运行。", file=sys.stderr)
            robo_mask = np.zeros((h, w), dtype=np.float32)
        finally:
            os.chdir(cwd_before)

    print(f"[dealbottle] 瓶子 mask 前景像素数: {int(bottle_mask.sum())}")

    # 4. 合并：机械臂 + 瓶子，再用于纹理增强
    mask = np.clip(robo_mask + bottle_mask, 0.0, 1.0).astype(np.float32)
    n_merged = int((mask > 0).sum())
    print(f"[dealbottle] 合并后前景像素数: {n_merged}（机械臂+瓶子）")

    out_mask_path = args.out_mask
    cv2.imwrite(out_mask_path, (mask * 255).astype(np.uint8))
    print(f"已保存合并 mask: {out_mask_path}")
    if args.debug:
        cv2.imwrite("robo_mask.png", (robo_mask * 255).astype(np.uint8))
        cv2.imwrite("bottle_mask.png", (bottle_mask * 255).astype(np.uint8))
        print("已保存调试用 robo_mask.png / bottle_mask.png")

    # 5. 纹理目录
    texture_root = Path(args.texture_dir) if args.texture_dir else Path(__file__).resolve().parent / "textures" / "mil_data"
    texture_paths = sorted(texture_root.glob("**/*.png"))
    if not texture_paths:
        raise FileNotFoundError(
            f"No texture PNGs under: {texture_root}\n"
            "Please download eugeneteoh/mil_data (png) to that folder, or set --texture-dir."
        )

    # 6. 用「合并后的 mask」做纹理背景增强：前景（机械臂+瓶子）保留原图，其余替换为纹理
    aug_images = []
    m_hwc = mask[:, :, None].astype(np.float32)  # (H,W,1)，与 img_rgb 一致
    for i in range(args.num_aug):
        bg_path = random.choice(texture_paths)
        bg_bgr = cv2.imread(str(bg_path), cv2.IMREAD_UNCHANGED)
        if bg_bgr is None:
            raise RuntimeError(f"Failed to read: {bg_path}")
        if bg_bgr.ndim == 2:
            bg_bgr = cv2.cvtColor(bg_bgr, cv2.COLOR_GRAY2BGR)
        elif bg_bgr.shape[2] == 4:
            bg_bgr = cv2.cvtColor(bg_bgr, cv2.COLOR_BGRA2BGR)
        bg_bgr = cv2.resize(bg_bgr, (w, h), interpolation=cv2.INTER_LINEAR)
        bg_rgb = cv2.cvtColor(bg_bgr, cv2.COLOR_BGR2RGB)

        aug_img = (img_rgb.astype(np.float32) * m_hwc + bg_rgb.astype(np.float32) * (1.0 - m_hwc)).astype(np.uint8)
        aug_images.append(aug_img)

        aug_bgr = cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR)
        file_name = Path(mask_path_abs).name
        pic_name = os.path.splitext(file_name)[0]
        out_name = f"{pic_name}_{i}.png"
        cv2.imwrite(out_name, aug_bgr)
        print(f"已保存: {out_name}")

    return aug_images


if __name__ == "__main__":
    main()
