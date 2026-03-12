import os
import json
import cv2
import numpy as np
import click
from loguru import logger
import matplotlib.pyplot as plt

# 设置 OpenCV 使用的后端，避免 Qt 插件问题
os.environ['QT_QPA_PLATFORM'] = 'offscreen'


class AnnotationRestorer:
    """标注恢复和可视化类"""

    def __init__(self, annotation_json_path, image_dir, mapping_json_path):
        """初始化

        Args:
            annotation_json_path: 标注结果 JSON 文件路径
            image_dir: 原始图片目录
            mapping_json_path: images2video 生成的映射关系 JSON 文件
        """
        self.annotation_json_path = annotation_json_path
        self.image_dir = image_dir
        self.mapping_json_path = mapping_json_path
        self.rng = np.random.default_rng()  # 使用新的随机数生成器

        # 加载数据
        self._load_annotations()
        self._load_mapping()

    def _load_annotations(self):
        """加载标注数据"""
        with open(self.annotation_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.objs = data['objs']  # 实例 ID -> 帧索引 -> pts
            self.frame_obj = data.get('frame_obj', {})  # 帧索引 -> 实例 ID 列表

        # 转换为帧索引 -> 实例数据的格式
        self.annotations = self._convert_to_frame_based()
        logger.info(f"加载标注数据: {len(self.annotations)} 帧")

    def _convert_to_frame_based(self):
        """将实例为主的数据转换为帧为主的数据

        Returns:
            帧索引 -> {instance_id: {pts: [...]}}
        """
        frame_based = {}

        for instance_id, frames_data in self.objs.items():
            for frame_index, frame_data in frames_data.items():
                logger.info(f"frame_index: {frame_index}")
                if frame_index == 'frame_list':
                    continue

                frame_index_int = int(frame_index)

                if frame_index_int not in frame_based:
                    frame_based[frame_index_int] = {}

                frame_based[frame_index_int][instance_id] = {
                    'pts': frame_data.get('pts', [])
                }

        return frame_based

    def _load_mapping(self):
        """加载映射关系"""
        with open(self.mapping_json_path, 'r', encoding='utf-8') as f:
            self.mapping_data = json.load(f)
        logger.info(f"加载映射数据: {len(self.mapping_data['frames'])} 帧")

        # 创建 frame_index 到 frame_info 的映射
        self.frame_index_to_info = {}
        for frame_info in self.mapping_data['frames']:
            frame_index = frame_info['frame_index']
            self.frame_index_to_info[frame_index] = frame_info
    def _draw_contours_on_image(self, image, contours, color=(0, 255, 0), alpha=0.5):
        """在图片上绘制轮廓

        Args:
            image: 输入图片
            contours: 轮廓点列表，格式为 [[[x, y]], [[x, y]], ...]
            color: 轮廓颜色
            alpha: 透明度

        Returns:
            叠加后的图片
        """
        if not contours:
            return image

        overlay = image.copy()

        # 转换为 numpy 数组，contours 格式: [[[x, y]], [[x, y]], ...]
        pts = np.array(contours, dtype=np.int32)
        # 绘制填充多边形
        cv2.fillPoly(overlay, [pts], color)

        # 叠加
        result = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
        return result

    def _get_original_image(self, frame_index):
        """获取原始图片

        Args:
            frame_index: 帧索引

        Returns:
            原始图片
        """
        if frame_index not in self.frame_index_to_info:
            return None

        frame_info = self.frame_index_to_info[frame_index]
        original_image_name = frame_info['frame_name']
        image_path = os.path.join(self.image_dir, original_image_name)

        if not os.path.exists(image_path):
            logger.warning(f"图片不存在: {image_path}")
            return None

        image = cv2.imread(image_path)
        return image

    def _transform_contours_to_original(self, contours_1920, frame_index):
        """将 1920 分辨率下的轮廓转换到原始分辨率

        Args:
            contours_1920: 1920 分辨率下的轮廓列表
            frame_index: 帧索引

        Returns:
            原始分辨率下的轮廓列表
        """
        if frame_index not in self.frame_index_to_info:
            return []

        frame_info = self.frame_index_to_info[frame_index]
        original_width = frame_info['original_size']['width']
        original_height = frame_info['original_size']['height']
        resized_width = frame_info['resized_size']['width']
        resized_height = frame_info['resized_size']['height']
        x_offset = frame_info['offset']['x']
        y_offset = frame_info['offset']['y']

        # 计算缩放比例
        scale_x = original_width / resized_width
        scale_y = original_height / resized_height

        # contours_1920 格式: [[[x, y]], [[x, y]], ...]
        contours_original = []
        for pt in contours_1920:
            x_1920, y_1920 = pt[0]  # pt 是 [[x, y]]
            # 减去偏移量
            x_resized = x_1920 - x_offset
            y_resized = y_1920 - y_offset
            # 缩放到原始尺寸
            x_original = int(x_resized * scale_x)
            y_original = int(y_resized * scale_y)
            contours_original.append([[x_original, y_original]])

        return contours_original

    def _draw_masks_1920(self, canvas_1920, instances):
        """在 1920 分辨率下绘制 mask

        Args:
            canvas_1920: 1920x1080 画布
            instances: 实例标注数据

        Returns:
            绘制后的图片
        """
        vis_1920 = canvas_1920.copy()
        for instance_id, instance_data in instances.items():
            contours = instance_data.get('pts', [])
            if contours:
                # 为每个实例生成固定颜色（基于 instance_id）
                color = self._get_instance_color(instance_id)
                vis_1920 = self._draw_contours_on_image(vis_1920, contours, color=color)
        return vis_1920

    def _draw_masks_original(self, original_image, instances, frame_index):
        """在原始分辨率下绘制 mask

        Args:
            original_image: 原始图片
            instances: 实例标注数据
            frame_index: 帧索引

        Returns:
            绘制后的图片
        """
        vis_original = original_image.copy()
        for instance_id, instance_data in instances.items():
            contours_1920 = instance_data.get('pts', [])
            if contours_1920:
                contours_original = self._transform_contours_to_original(contours_1920, frame_index)
                # 为每个实例生成固定颜色（基于 instance_id）
                color = self._get_instance_color(instance_id)
                vis_original = self._draw_contours_on_image(vis_original, contours_original, color=color)
        return vis_original

    def _get_instance_color(self, instance_id):
        """为实例 ID 生成固定颜色

        Args:
            instance_id: 实例 ID

        Returns:
            RGB 颜色元组
        """
        # 使用 hash 生成固定颜色
        hash_value = hash(str(instance_id))
        r = (hash_value & 0xFF0000) >> 16
        g = (hash_value & 0x00FF00) >> 8
        b = hash_value & 0x0000FF
        # 确保颜色不太暗
        r = max(r, 100)
        g = max(g, 100)
        b = max(b, 100)
        return (b, g, r)  # OpenCV 使用 BGR

    def visualize(self):
        """可视化标注结果，支持按键切换帧"""

        frame_indices = sorted(self.annotations.keys())
        if not frame_indices:
            logger.error("没有标注数据")
            return

        current_idx = 0
        logger.info("按键说明: 's' 下一帧, 'w' 上一帧, 'q' 退出")

        while True:
            frame_index = frame_indices[current_idx]

            logger.info("=" * 60)
            logger.info(f"当前索引: {current_idx}/{len(frame_indices)-1}")
            logger.info(f"当前帧索引: {frame_index}")

            # 获取原始图片
            original_image = self._get_original_image(frame_index)
            if original_image is None:
                logger.warning(f"跳过帧 {frame_index}")
                current_idx = (current_idx + 1) % len(frame_indices)
                continue

            # 获取 1920 分辨率下的图片
            frame_info = self.frame_index_to_info[frame_index]
            logger.info(f"原始图片名: {frame_info['original_image_name']}")
            logger.info(f"原始尺寸: {frame_info['original_size']}")
            logger.info(f"缩放尺寸: {frame_info['resized_size']}")
            logger.info(f"偏移量: {frame_info['offset']}")
            resized_width = frame_info['resized_size']['width']
            resized_height = frame_info['resized_size']['height']
            x_offset = frame_info['offset']['x']
            y_offset = frame_info['offset']['y']
            target_width = self.mapping_data['target_size']['width']
            target_height = self.mapping_data['target_size']['height']

            # 缩放原始图片到 1920 分辨率
            resized_image = cv2.resize(original_image, (resized_width, resized_height))
            canvas_1920 = np.zeros((target_height, target_width, 3), dtype=np.uint8)
            canvas_1920[y_offset:y_offset + resized_height, x_offset:x_offset + resized_width] = resized_image

            # 获取标注数据
            instances = self.annotations.get(frame_index, {})
            logger.info(f"标注实例数: {len(instances)}")
            if instances:
                logger.info(f"实例ID列表: {list(instances.keys())}")
                for inst_id, inst_data in instances.items():
                    pts_count = len(inst_data.get('pts', []))
                    logger.info(f"  实例 {inst_id}: {pts_count} 个点")

            # 绘制 1920 分辨率下的 mask
            vis_1920 = self._draw_masks_1920(canvas_1920, instances)

            # 绘制原始分辨率下的 mask
            vis_original = self._draw_masks_original(original_image, instances, frame_index)

            # 添加文本信息
            info_text_1920 = f"Frame {frame_index} | Instances: {len(instances)} | {current_idx+1}/{len(frame_indices)}"
            info_text_orig = f"Frame {frame_index} | {frame_info['original_image_name']}"

            cv2.putText(vis_1920, info_text_1920, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(vis_1920, "1920x1080", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

            cv2.putText(vis_original, info_text_orig, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            orig_size_text = f"{frame_info['original_size']['width']}x{frame_info['original_size']['height']}"
            cv2.putText(vis_original, orig_size_text, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

            # 显示
            cv2.imshow("1920x1080 Resolution", vis_1920)
            cv2.imshow("Original Resolution", vis_original)

            # 等待按键
            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):  # 下一帧
                current_idx = (current_idx + 1) % len(frame_indices)
            elif key == ord('w'):  # 上一帧
                current_idx = (current_idx - 1) % len(frame_indices)

        cv2.destroyAllWindows()
        logger.info("可视化结束")

    def restore_and_save(self, output_dir):
        """恢复标注到原始分辨率并保存

        Args:
            output_dir: 输出目录
        """
        os.makedirs(output_dir, exist_ok=True)
        logger.info("开始恢复标注到原始分辨率...")

        restored_annotations = {}

        for frame_index, instances in self.annotations.items():
            if frame_index not in self.frame_index_to_info:
                logger.warning(f"跳过帧 {frame_index}: 映射信息不存在")
                continue

            frame_info = self.frame_index_to_info[frame_index]
            original_image_name = frame_info['frame_name']

            restored_instances = {}
            for instance_id, instance_data in instances.items():
                contours_1920 = instance_data.get('pts', [])
                if contours_1920:
                    # 转换到原始分辨率
                    contours_original = self._transform_contours_to_original(contours_1920, frame_index)
                    restored_instances[instance_id] = {
                        'pts': contours_original
                    }

            if restored_instances:
                restored_annotations[original_image_name] = restored_instances

        # 保存恢复后的标注
        output_json_path = os.path.join(output_dir, 'restored_annotations.json')
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(restored_annotations, f, indent=2, ensure_ascii=False)

        logger.info("恢复完成！")
        logger.info(f"恢复了 {len(restored_annotations)} 张图片的标注")
        logger.info(f"输出文件: {output_json_path}")


def vis_image_from_restored_annotations(restored_annotations_path, image_dir):
    """可视化恢复后的标注结果

    Args:
        restored_annotations_path: 恢复后的标注 JSON 文件路径
        image_dir: 原始图片目录
    """
    with open(restored_annotations_path, 'r', encoding='utf-8') as f:
        restored_annotations = json.load(f)

    logger.info(f"加载了 {len(restored_annotations)} 张图片的标注")

    for image_name, instances in restored_annotations.items():
        image_path = os.path.join(image_dir, image_name)
        image = cv2.imread(image_path)

        if image is None:
            logger.warning(f"无法读取图片: {image_path}")
            continue

        logger.info(f"图片: {image_name}, 实例数: {len(instances)}")

        # 转换为 RGB 用于 matplotlib 显示
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 为每个实例生成固定颜色
        for instance_id, instance_data in instances.items():
            contours = instance_data.get('pts', [])
            if contours:
                # 转换为 numpy 数组，格式: [[[x, y]], [[x, y]], ...]
                pts = np.array(contours, dtype=np.int32)

                # 生成固定颜色 (RGB 格式用于 matplotlib)
                rgba = plt.cm.tab20(hash(str(instance_id)) % 20)
                color_rgb = (int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255))

                # 绘制填充多边形
                overlay = image_rgb.copy()
                cv2.fillPoly(overlay, [pts], color_rgb)
                image_rgb = cv2.addWeighted(overlay, 0.5, image_rgb, 0.5, 0)

                # 绘制轮廓
                cv2.polylines(image_rgb, [pts], True, color_rgb, 2)

                logger.info(f"  实例 {instance_id}: {len(contours)} 个点")

        os.makedirs(os.path.join(image_dir, "vis"), exist_ok=True)

        cv2.imwrite(os.path.join(image_dir, "vis",  image_name), image_rgb)

    logger.info("可视化完成！")


@click.command()
@click.option("--image_dir", "-i", type=str, required=True, help="原始图片目录")
def run(image_dir):
    mapping_json = os.path.join(image_dir, "video_1920_1080_mapping.json")

    json_path = os.path.join(image_dir, "video_1920_1080_anno.json")

    restorer = AnnotationRestorer(
        annotation_json_path=json_path,
        image_dir=image_dir,
        mapping_json_path=mapping_json
    )

    # 可视化
    # restorer.visualize()

    restorer.restore_and_save(image_dir)

    vis_image_from_restored_annotations(os.path.join(image_dir, "restored_annotations.json"), image_dir)


if __name__ == "__main__":

    #image_dir = "C:\\Users\\26370\\Desktop\\bottle2_data\\bottle2"
    run()
