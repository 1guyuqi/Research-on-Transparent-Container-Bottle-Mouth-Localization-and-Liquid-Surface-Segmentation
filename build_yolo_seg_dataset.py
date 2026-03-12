import argparse
import json
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


IMG_W = 1920
IMG_H = 1080


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class SplitRatios:
    train: float
    val: float
    test: float


def _is_number(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


def flatten_pts(raw_pts) -> List[Tuple[float, float]]:
    """
    兼容以下两种常见格式：
    - [[[x,y]], [[x,y]], ...]  （你这个 json 里就是这种）
    - [[x,y], [x,y], ...]
    """
    pts: List[Tuple[float, float]] = []
    if not isinstance(raw_pts, list):
        return pts

    for item in raw_pts:
        if not isinstance(item, list) or len(item) == 0:
            continue

        # case A: [[x, y]]
        if (
            len(item) == 1
            and isinstance(item[0], list)
            and len(item[0]) >= 2
            and _is_number(item[0][0])
            and _is_number(item[0][1])
        ):
            x, y = float(item[0][0]), float(item[0][1])
        # case B: [x, y]
        elif len(item) >= 2 and _is_number(item[0]) and _is_number(item[1]):
            x, y = float(item[0]), float(item[1])
        else:
            continue

        x = _clamp(x, 0.0, float(IMG_W))
        y = _clamp(y, 0.0, float(IMG_H))
        pts.append((x, y))

    return pts


def bbox_from_polygon(pts: Sequence[Tuple[float, float]]) -> Optional[Tuple[float, float, float, float]]:
    if len(pts) < 3:
        return None

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    w = x_max - x_min
    h = y_max - y_min
    if w <= 0 or h <= 0:
        return None

    x_c = x_min + w / 2.0
    y_c = y_min + h / 2.0

    return (x_c / IMG_W, y_c / IMG_H, w / IMG_W, h / IMG_H)


def normalize_polygon(pts: Sequence[Tuple[float, float]]) -> List[float]:
    out: List[float] = []
    for x, y in pts:
        out.append(x / IMG_W)
        out.append(y / IMG_H)
    return out


def list_images(images_dir: Path) -> List[Path]:
    imgs: List[Path] = []
    for p in images_dir.iterdir():
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            imgs.append(p)
    imgs.sort(key=lambda x: x.name)
    return imgs


def frame_id_from_image_name(img_path: Path) -> Optional[int]:
    """
    适配 000001.jpg / 1.jpg / xxx_000001.jpg 等：
    - 优先：文件名去掉后缀后全是数字
    - 否则：取 basename 中最后一段连续数字作为 frame_id
    """
    stem = img_path.stem
    if stem.isdigit():
        return int(stem)
    m = re.search(r"(\d+)(?!.*\d)", stem)
    if not m:
        return None
    return int(m.group(1))


def safe_copy(src: Path, dst: Path, overwrite: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and not overwrite:
        return
    import shutil

    shutil.copy2(src, dst)


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def compute_splits(items: Sequence[Path], ratios: SplitRatios, seed: int) -> Dict[str, List[Path]]:
    items = list(items)
    rnd = random.Random(seed)
    rnd.shuffle(items)

    n = len(items)

    # 允许用户显式把 test 设为 0，此时保证不产生 test 样本，余数分配给 train
    r_train, r_val, r_test = ratios.train, ratios.val, ratios.test
    if r_test == 0.0:
        n_train = int(round(n * r_train))
        n_val = n - n_train
        n_test = 0
    else:
        n_train = int(round(n * r_train))
        n_val = int(round(n * r_val))
        n_test = n - n_train - n_val
        if n_test < 0:
            # 超了就从 train 回收，保证不为负
            n_train = max(0, n_train + n_test)
            n_test = 0

    train = items[:n_train]
    val = items[n_train : n_train + n_val]
    test = items[n_train + n_val : n_train + n_val + n_test]
    return {"train": train, "val": val, "test": test}


def build_names_list(attributes: Dict[str, str]) -> List[str]:
    """
    attributes 是 {"0": "试管", ...}。
    YOLO 需要 names 为 0..nc-1 的列表；这里按最大 id 补齐。
    """
    ids: List[int] = []
    for k in attributes.keys():
        try:
            ids.append(int(k))
        except ValueError:
            continue
    if not ids:
        return []

    max_id = max(ids)
    names = [f"class_{i}" for i in range(max_id + 1)]
    for k, v in attributes.items():
        try:
            i = int(k)
        except ValueError:
            continue
        if 0 <= i < len(names):
            names[i] = str(v)
    return names


def main():
    parser = argparse.ArgumentParser(description="Build YOLO-seg dataset from video_1920_1080_anno.json and images/")
    parser.add_argument("--root", default=".", help="prp 目录（包含 images/ 和 video_1920_1080_anno.json）")
    parser.add_argument("--json", default="video_1920_1080_anno.json", help="标注 json 文件名或路径")
    parser.add_argument("--images", default="images", help="图片目录名或路径")
    parser.add_argument("--out", default="yolo_seg_dataset", help="输出数据集目录（会在 root 下创建）")
    parser.add_argument("--seed", type=int, default=42, help="划分数据集随机种子")
    parser.add_argument("--img-w", type=int, default=1920, help="图片宽度（用于坐标归一化）")
    parser.add_argument("--img-h", type=int, default=1080, help="图片高度（用于坐标归一化）")
    parser.add_argument("--train", type=float, default=0.8, help="train 比例")
    parser.add_argument("--val", type=float, default=0.1, help="val 比例")
    parser.add_argument("--test", type=float, default=0.1, help="test 比例")
    parser.add_argument(
        "--copy",
        action="store_true",
        help="复制图片文件本体到数据集目录（默认行为就是复制；保留该开关是为了兼容旧命令）",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="若输出目录已存在则先删除再重新生成（会删除 out 目录，请谨慎）",
    )
    parser.add_argument(
        "--label-format",
        choices=["ultralytics", "polyonly"],
        default="ultralytics",
        help="标签行格式：ultralytics=class bbox poly；polyonly=class poly（不含 bbox，可能不被 YOLOv8-seg 接受）",
    )
    parser.add_argument(
        "--minus1",
        choices=["skip", "keep", "map"],
        default="skip",
        help='attri == "-1" 的处理：skip=忽略；keep=写 -1（不保证可训练）；map=映射到新类 unknown',
    )
    args = parser.parse_args()

    # 用命令行参数覆盖默认宽高（脚本内部使用全局常量）
    global IMG_W, IMG_H
    IMG_W = int(args.img_w)
    IMG_H = int(args.img_h)

    root = Path(args.root).resolve()
    json_path = (root / args.json).resolve() if not os.path.isabs(args.json) else Path(args.json).resolve()
    images_dir = (root / args.images).resolve() if not os.path.isabs(args.images) else Path(args.images).resolve()
    out_root = (root / args.out).resolve()

    if args.force and out_root.exists():
        import shutil

        shutil.rmtree(out_root)

    data = json.loads(json_path.read_text(encoding="utf-8"))

    frame_obj = data.get("frame_obj", {})
    objs = data.get("objs", {})
    attributes = (data.get("label_cfg", {}) or {}).get("attributes", {})

    # 预扫描：是否存在可用的非 -1 类别（attri 在 attributes 中）
    has_known_class = False
    if args.minus1 == "map":
        for obj_entry in objs.values():
            if not isinstance(obj_entry, dict):
                continue
            for k, v in obj_entry.items():
                if k == "frame_list":
                    continue
                if not isinstance(v, dict):
                    continue
                attri_val = str(v.get("attri", "-1"))
                if attri_val != "-1" and attri_val in attributes:
                    has_known_class = True
                    break
            if has_known_class:
                break

    names = build_names_list(attributes)
    base_nc = len(names)

    # minus1 == map：
    # - 若有已知类别：unknown 追加到末尾（id=base_nc），不影响已有 id
    # - 若全部都是 -1：压缩成单类 unknown（id=0），更适合直接训练
    compact_unknown_only = args.minus1 == "map" and not has_known_class
    if args.minus1 == "map":
        if compact_unknown_only:
            names = ["unknown"]
        else:
            names = list(names) + ["unknown"]

    ratios = SplitRatios(train=args.train, val=args.val, test=args.test)

    imgs = list_images(images_dir)
    splits = compute_splits(imgs, ratios, args.seed)

    # 生成目录
    for split_name in ("train", "val", "test"):
        if split_name == "test" and ratios.test == 0.0:
            continue
        (out_root / "images" / split_name).mkdir(parents=True, exist_ok=True)
        (out_root / "labels" / split_name).mkdir(parents=True, exist_ok=True)

    stats = {
        "images_total": len(imgs),
        "splits": {k: len(v) for k, v in splits.items()},
        "labels_written": 0,
        "instances_written": 0,
        "instances_skipped_minus1": 0,
        "instances_skipped_no_attr_mapping": 0,
        "instances_skipped_invalid_polygon": 0,
        "images_without_frame_id": 0,
    }

    def obj_ids_for_frame(frame_id: int) -> List[int]:
        raw = frame_obj.get(str(frame_id), [])
        if not isinstance(raw, list):
            return []
        out: List[int] = []
        for x in raw:
            try:
                out.append(int(x))
            except Exception:
                continue
        return out

    for split_name, split_imgs in splits.items():
        if split_name == "test" and ratios.test == 0.0:
            continue
        for img_path in split_imgs:
            frame_id = frame_id_from_image_name(img_path)
            if frame_id is None:
                stats["images_without_frame_id"] += 1
                continue

            # 图片放到 out_root/images/<split>/ 下（复制文件本体）
            out_img = out_root / "images" / split_name / img_path.name
            safe_copy(img_path, out_img, overwrite=False)

            # 标签文件与图片同名（仅后缀换成 .txt）
            label_name = f"{img_path.stem}.txt"
            out_label = out_root / "labels" / split_name / label_name

            lines: List[str] = []

            for obj_id in obj_ids_for_frame(frame_id):
                obj_entry = objs.get(str(obj_id))
                if not isinstance(obj_entry, dict):
                    continue
                frame_info = obj_entry.get(str(frame_id))
                if not isinstance(frame_info, dict):
                    continue

                raw_pts = frame_info.get("pts", [])
                pts = flatten_pts(raw_pts)
                if len(pts) < 3:
                    stats["instances_skipped_invalid_polygon"] += 1
                    continue

                bbox = None
                if args.label_format == "ultralytics":
                    bbox = bbox_from_polygon(pts)
                    if bbox is None:
                        stats["instances_skipped_invalid_polygon"] += 1
                        continue

                attri_val = str(frame_info.get("attri", "-1"))
                if attri_val == "-1":
                    if args.minus1 == "skip":
                        stats["instances_skipped_minus1"] += 1
                        continue
                    if args.minus1 == "keep":
                        class_id = -1
                    else:  # map
                        class_id = 0 if compact_unknown_only else base_nc  # unknown id
                else:
                    if attri_val in attributes:
                        if compact_unknown_only:
                            # unknown-only 模式下，忽略已知类别（理论上不会出现）
                            stats["instances_skipped_no_attr_mapping"] += 1
                            continue
                        class_id = int(attri_val)
                    else:
                        stats["instances_skipped_no_attr_mapping"] += 1
                        continue

                poly = normalize_polygon(pts)
                if args.label_format == "polyonly":
                    parts = [str(class_id)] + [f"{v:.6f}" for v in poly]
                else:
                    x_c, y_c, w, h = bbox
                    parts = [
                        str(class_id),
                        f"{x_c:.6f}",
                        f"{y_c:.6f}",
                        f"{w:.6f}",
                        f"{h:.6f}",
                    ] + [f"{v:.6f}" for v in poly]

                lines.append(" ".join(parts))

            write_text(out_label, "\n".join(lines))
            stats["labels_written"] += 1
            stats["instances_written"] += len(lines)

    # data.yaml（Ultralytics）
    yaml_lines = [
        f"path: {out_root.as_posix()}",
        "train: images/train",
        "val: images/val",
        f"nc: {len(names)}",
        "names:",
    ]
    if ratios.test != 0.0:
        yaml_lines.insert(4, "test: images/test")
    for i, n in enumerate(names):
        yaml_lines.append(f"  {i}: {json.dumps(n, ensure_ascii=False)}")

    write_text(out_root / "data.yaml", "\n".join(yaml_lines) + "\n")
    write_text(out_root / "dataset_stats.json", json.dumps(stats, ensure_ascii=False, indent=2) + "\n")

    readme = f"""\
## YOLO-Seg 数据集

本数据集由 `{json_path.name}` + `images/` 自动生成。

### 目录结构

- `images/train|val|test/`：图片（默认软链接到原始 `images/`）
- `labels/train|val|test/`：YOLO-Seg 标签（与图片同名 `.txt`）
- `data.yaml`：Ultralytics 训练配置
- `dataset_stats.json`：生成统计

### 标签格式（每行一个实例）

本次生成使用 `--label-format {args.label_format}`：

- `ultralytics`：`class_id x_center y_center w h x1 y1 x2 y2 ... xn yn`
- `polyonly`：`class_id x1 y1 x2 y2 ... xn yn`（不含 bbox，可能不被 YOLOv8-seg 接受）

坐标均已按 W={IMG_W}, H={IMG_H} 归一化到 [0,1]。

### 关于 attri == -1

本次生成使用参数 `--minus1 {args.minus1}`：

- `skip`：忽略这些实例（推荐，保证可训练）
- `map`：把 -1 映射成新类 `unknown`
- `keep`：写 -1（Ultralytics 训练通常不接受负类 id，不保证可训练）

### 训练命令示例（YOLOv8-seg）

```bash
yolo segment train data={out_root.as_posix()}/data.yaml model=yolov8n-seg.pt
```
"""
    write_text(out_root / "README.md", readme)

    print("Done.")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"Dataset root: {out_root}")


if __name__ == "__main__":
    main()

