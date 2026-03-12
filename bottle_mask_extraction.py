# -*- coding: utf-8 -*-
"""
YOLOv11 分割模型 ONNX 推理脚本 —— 输出化学容器 mask
使用方式: 修改下方 IMAGE_PATH 为你的图片路径，然后运行 python yolo_seg_infer.py
"""

import math
import argparse
import numpy as np
import cv2
import onnxruntime as ort

# ============ 默认路径（可修改） ============
MODEL_PATH = r"C:\Users\26370\Desktop\yolov11_seg_1129_640_480.onnx"
IMAGE_PATH = r"C:\Users\26370\Desktop\roboengine-main\roboengine-main\results1\bottle_image1.jpg"  # 请改为你的图片路径
OUTPUT_MASK_PATH = r"C:\Users\26370\Desktop\container_mask.png"
OUTPUT_OVERLAY_PATH = r"C:\Users\26370\Desktop\container_overlay.png"
NUM_MASKS = 32
CONF_THRESH = 0.25
IOU_THRESH = 0.5


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def xywh2xyxy(x):
    """(x_center, y_center, w, h) -> (x1, y1, x2, y2)"""
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def nms(boxes, scores, iou_threshold):
    x1, y1 = boxes[:, 0], boxes[:, 1]
    x2, y2 = boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = np.argsort(scores)[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[np.where(iou <= iou_threshold)[0] + 1]
    return keep


def rescale_boxes(boxes_xywh, from_shape, to_shape):
    """from_shape/to_shape: (height, width)"""
    h_in, w_in = from_shape[0], from_shape[1]
    h_out, w_out = to_shape[0], to_shape[1]
    scale = np.array([w_in, h_in, w_in, h_in], dtype=np.float32)
    boxes_xywh = boxes_xywh.astype(np.float32) / scale
    boxes_xywh *= np.array([w_out, h_out, w_out, h_out], dtype=np.float32)
    return boxes_xywh


class YOLOv11Seg:
    def __init__(self, model_path, conf_thres=0.25, iou_thres=0.5, num_masks=32):
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.num_masks = num_masks
        self.session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],
        )
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        # 通常 shape 为 ['batch', 3, height, width]
        self.input_height = int(inp.shape[2])
        self.input_width = int(inp.shape[3])
        self.output_names = [o.name for o in self.session.get_outputs()]

    def preprocess(self, image_bgr):
        img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img_rgb, (self.input_width, self.input_height))
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)  # HWC -> CHW
        blob = img[np.newaxis, ...].astype(np.float32)
        return blob

    def run(self, image_bgr):
        self.img_h, self.img_w = image_bgr.shape[:2]
        blob = self.preprocess(image_bgr)
        outputs = self.session.run(self.output_names, {self.input_name: blob})
        # output0: (1, 4+num_classes+32, N), output1: (1, 32, mask_h, mask_w)
        box_out = outputs[0]
        proto_out = outputs[1]
        boxes, scores, class_ids, mask_coeffs = self._parse_detections(box_out)
        if len(boxes) == 0:
            return boxes, scores, class_ids, []
        mask_maps = self._decode_masks(mask_coeffs, proto_out, boxes)
        return boxes, scores, class_ids, mask_maps

    def _parse_detections(self, box_output):
        # box_output: (1, C, N)，C = 4 + num_classes + 32
        raw = np.squeeze(box_output)  # (C, N)
        if raw.ndim == 1:
            raw = raw[:, np.newaxis]
        pred = raw.T  # (N, C)
        num_classes = pred.shape[1] - 4 - self.num_masks
        if num_classes <= 0:
            num_classes = 1
        # 前 4 为 bbox(xywh)，接着 num_classes 为类别得分，最后 32 为 mask 系数
        class_scores = pred[:, 4 : 4 + num_classes]
        scores = np.max(class_scores, axis=1)
        keep = scores >= self.conf_thres
        if not np.any(keep):
            return np.array([]), np.array([]), np.array([]), np.array([])
        pred = pred[keep]
        scores = scores[keep]
        class_ids = np.argmax(pred[:, 4 : 4 + num_classes], axis=1)
        boxes_xywh = pred[:, :4]
        boxes_xywh = rescale_boxes(
            boxes_xywh,
            (self.input_height, self.input_width),
            (self.img_h, self.img_w),
        )
        boxes = xywh2xyxy(boxes_xywh)
        boxes[:, 0] = np.clip(boxes[:, 0], 0, self.img_w)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, self.img_h)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, self.img_w)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, self.img_h)
        indices = nms(boxes, scores, self.iou_thres)
        boxes = boxes[indices]
        scores = scores[indices]
        class_ids = class_ids[indices]
        mask_coeffs = pred[indices, 4 + num_classes : 4 + num_classes + self.num_masks]
        return boxes, scores, class_ids, mask_coeffs

    def _decode_masks(self, mask_coeffs, proto_output, boxes):
        # proto_output: (1, 32, mask_h, mask_w)
        proto = np.squeeze(proto_output)  # (32, mask_h, mask_w)
        num_proto, mh, mw = proto.shape
        # mask_coeffs: (num_det, 32), proto flatten: (32, mh*mw)
        masks = sigmoid(mask_coeffs @ proto.reshape(num_proto, -1))
        masks = masks.reshape(-1, mh, mw)
        # 将每个 mask 裁到对应 box 并 resize 回原图（boxes 已是 xyxy，按原图尺寸缩放到 mask 尺寸）
        scale_boxes = rescale_boxes(
            boxes.copy(),
            (self.img_h, self.img_w),
            (mh, mw),
        )
        out_maps = np.zeros((len(boxes), self.img_h, self.img_w), dtype=np.float32)
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i].astype(int)
            sx1 = int(np.clip(np.floor(scale_boxes[i][0]), 0, mw - 1))
            sy1 = int(np.clip(np.floor(scale_boxes[i][1]), 0, mh - 1))
            sx2 = int(np.clip(np.ceil(scale_boxes[i][2]), 1, mw))
            sy2 = int(np.clip(np.ceil(scale_boxes[i][3]), 1, mh))
            crop = masks[i, sy1:sy2, sx1:sx2]
            if crop.size == 0:
                continue
            crop = cv2.resize(crop, (x2 - x1, y2 - y1), interpolation=cv2.INTER_LINEAR)
            crop = (crop > 0.5).astype(np.float32)
            out_maps[i, y1:y2, x1:x2] = crop
        return out_maps


def main():
    parser = argparse.ArgumentParser(description="YOLOv11 分割推理，输出容器 mask")
    parser.add_argument("--model", default=MODEL_PATH, help="ONNX 模型路径")
    parser.add_argument("--image", default=IMAGE_PATH, help="输入图片路径")
    parser.add_argument("--out-mask", default=OUTPUT_MASK_PATH, help="输出 mask 图路径（二值）")
    parser.add_argument("--out-overlay", default=OUTPUT_OVERLAY_PATH, help="输出叠加可视化图路径")
    parser.add_argument("--conf", type=float, default=CONF_THRESH, help="置信度阈值")
    parser.add_argument("--iou", type=float, default=IOU_THRESH, help="NMS IoU 阈值")
    args = parser.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        raise FileNotFoundError(f"找不到图片: {args.image}")

    model = YOLOv11Seg(args.model, conf_thres=args.conf, iou_thres=args.iou, num_masks=NUM_MASKS)
    boxes, scores, class_ids, mask_maps = model.run(img)

    # 合并所有实例的 mask 为一张二值 mask（容器区域为 255）
    if len(mask_maps) > 0:
        combined_mask = (np.max(mask_maps, axis=0) > 0.5).astype(np.uint8) * 255
    else:
        combined_mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        print("未检测到任何容器，mask 将保存为全黑图。")

    cv2.imwrite(args.out_mask, combined_mask)
    print(f"已保存二值 mask: {args.out_mask}")

    # 可选：保存叠加图（原图 + 半透明 mask）
    overlay = img.copy()
    if combined_mask.max() > 0:
        overlay[combined_mask > 0] = (
            overlay[combined_mask > 0] * 0.5 + np.array([0, 180, 0], dtype=np.uint8) * 0.5
        ).astype(np.uint8)
    cv2.imwrite(args.out_overlay, overlay)
    print(f"已保存叠加可视化: {args.out_overlay}")

    print(f"检测到 {len(boxes)} 个容器。")
    return combined_mask, mask_maps


if __name__ == "__main__":
    main()
