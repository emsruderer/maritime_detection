#!/usr/bin/env python3
"""
Hailo-8 Object Detection with YOLOv8n
Requires: hailo_platform, OpenCV, numpy
HEF model: yolov8n.hef (compile from Hailo Model Zoo or download)
"""

import cv2
import numpy as np
import time
from pathlib import Path

import hailo_platform

# ── Config ────────────────────────────────────────────────────────────────────
HEF_PATH      = "./models/yolov8n.hef"
VIDEO_SOURCE  = "/dev/video16"          # Logitech BRIO USB camera (use /dev/video17, /dev/video18, or /dev/video36 if this doesn't work)
CONF_THRESH   = 0.35
IOU_THRESH    = 0.45
INPUT_W       = 640
INPUT_H       = 640

COCO_CLASSES = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck",
    "boat","traffic light","fire hydrant","stop sign","parking meter","bench",
    "bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe",
    "backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard",
    "sports ball","kite","baseball bat","baseball glove","skateboard","surfboard",
    "tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl",
    "banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza",
    "donut","cake","chair","couch","potted plant","bed","dining table","toilet",
    "tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven",
    "toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear",
    "hair drier","toothbrush",
]

np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(COCO_CLASSES), 3), dtype=np.uint8)

# ── Preprocessing ─────────────────────────────────────────────────────────────

def preprocess(frame: np.ndarray) -> np.ndarray:
    """Letterbox resize → RGB → uint8 NHWC"""
    h, w = frame.shape[:2]
    scale = min(INPUT_W / w, INPUT_H / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((INPUT_H, INPUT_W, 3), 114, dtype=np.uint8)
    pad_x = (INPUT_W - nw) // 2
    pad_y = (INPUT_H - nh) // 2
    canvas[pad_y:pad_y+nh, pad_x:pad_x+nw] = resized

    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    return rgb.astype(np.uint8), scale, pad_x, pad_y


# ── Postprocessing ────────────────────────────────────────────────────────────

def xywh2xyxy(boxes):
    """Convert cx,cy,w,h → x1,y1,x2,y2"""
    b = boxes.copy()
    b[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    b[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    b[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    b[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    return b


def nms(boxes, scores, iou_thresh):
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        area_i = (boxes[i, 2]-boxes[i, 0]) * (boxes[i, 3]-boxes[i, 1])
        area_o = (boxes[order[1:], 2]-boxes[order[1:], 0]) * \
                 (boxes[order[1:], 3]-boxes[order[1:], 1])
        iou = inter / (area_i + area_o - inter + 1e-6)
        order = order[1:][iou <= iou_thresh]
    return keep


def postprocess(raw_output: np.ndarray, scale, pad_x, pad_y,
                orig_w, orig_h, conf_thresh=CONF_THRESH, iou_thresh=IOU_THRESH):
    """
    Supports:
    - Raw YOLOv8 tensor: (1, 84, 8400)
    - Hailo NMS output: (1, classes, detections, 5) or (classes, detections, 5)
      where 5 = [ymin, xmin, ymax, xmax, score]
    """
    if isinstance(raw_output, dict):
        raw_output = next(iter(raw_output.values()))
    if isinstance(raw_output, (list, tuple)):
        raw_output = raw_output[0] if len(raw_output) == 1 else raw_output
        if isinstance(raw_output, (list, tuple)):
            boxes_list = []
            confs_list = []
            cls_list = []

            for class_id, class_dets in enumerate(raw_output):
                class_arr = np.asarray(class_dets)
                if class_arr.size == 0:
                    continue
                class_arr = class_arr.reshape(-1, class_arr.shape[-1])

                for det in class_arr:
                    ymin, xmin, ymax, xmax, conf = det[:5]
                    conf = float(conf)
                    if conf < conf_thresh:
                        continue

                    if max(abs(xmin), abs(ymin), abs(xmax), abs(ymax)) <= 1.5:
                        x1 = float(xmin) * orig_w
                        y1 = float(ymin) * orig_h
                        x2 = float(xmax) * orig_w
                        y2 = float(ymax) * orig_h
                    else:
                        x1 = float(xmin)
                        y1 = float(ymin)
                        x2 = float(xmax)
                        y2 = float(ymax)

                    x1 = max(0.0, min(float(orig_w), x1))
                    x2 = max(0.0, min(float(orig_w), x2))
                    y1 = max(0.0, min(float(orig_h), y1))
                    y2 = max(0.0, min(float(orig_h), y2))
                    if x2 <= x1 or y2 <= y1:
                        continue

                    boxes_list.append([x1, y1, x2, y2])
                    confs_list.append(conf)
                    cls_list.append(class_id)

            if not boxes_list:
                return [], [], []

            boxes_xyxy = np.asarray(boxes_list, dtype=np.float32)
            confidences = np.asarray(confs_list, dtype=np.float32)
            class_ids = np.asarray(cls_list, dtype=np.int32)

            keep = nms(boxes_xyxy, confidences, iou_thresh)
            return boxes_xyxy[keep].astype(int), confidences[keep], class_ids[keep]

        raw_output = np.asarray(raw_output)
    else:
        raw_output = np.asarray(raw_output)

    # Hailo NMS path: class-wise boxes [ymin, xmin, ymax, xmax, score]
    nms_out = raw_output
    if nms_out.ndim == 4 and nms_out.shape[0] == 1:
        nms_out = nms_out[0]
    # Some HEFs expose NMS as (classes, 5, detections) instead of
    # (classes, detections, 5). Normalize to the latter.
    if nms_out.ndim == 3 and nms_out.shape[1] == 5 and nms_out.shape[-1] != 5:
        nms_out = np.transpose(nms_out, (0, 2, 1))
    if nms_out.ndim == 3 and nms_out.shape[-1] >= 5:
        boxes_list = []
        confs_list = []
        cls_list = []

        for class_id, class_dets in enumerate(nms_out):
            class_arr = np.asarray(class_dets)
            if class_arr.size == 0:
                continue
            class_arr = class_arr.reshape(-1, class_arr.shape[-1])

            for det in class_arr:
                ymin, xmin, ymax, xmax, conf = det[:5]
                conf = float(conf)
                if conf < conf_thresh:
                    continue

                if max(abs(xmin), abs(ymin), abs(xmax), abs(ymax)) <= 1.5:
                    x1 = float(xmin) * orig_w
                    y1 = float(ymin) * orig_h
                    x2 = float(xmax) * orig_w
                    y2 = float(ymax) * orig_h
                else:
                    x1 = float(xmin)
                    y1 = float(ymin)
                    x2 = float(xmax)
                    y2 = float(ymax)

                x1 = max(0.0, min(float(orig_w), x1))
                x2 = max(0.0, min(float(orig_w), x2))
                y1 = max(0.0, min(float(orig_h), y1))
                y2 = max(0.0, min(float(orig_h), y2))
                if x2 <= x1 or y2 <= y1:
                    continue

                boxes_list.append([x1, y1, x2, y2])
                confs_list.append(conf)
                cls_list.append(class_id)

        if not boxes_list:
            return [], [], []

        boxes_xyxy = np.asarray(boxes_list, dtype=np.float32)
        confidences = np.asarray(confs_list, dtype=np.float32)
        class_ids = np.asarray(cls_list, dtype=np.int32)

        keep = nms(boxes_xyxy, confidences, iou_thresh)
        return boxes_xyxy[keep].astype(int), confidences[keep], class_ids[keep]

    out = raw_output[0] if raw_output.ndim == 3 else raw_output

    if out.ndim != 2 or out.shape[0] < 5 or out.shape[1] == 0:
        return [], [], []

    boxes_raw = out[:4, :].T     # (8400, 4)  cx,cy,w,h  (in [0,1] or pixel space)
    scores_raw = out[4:, :].T    # (8400, 80)

    if scores_raw.shape[1] == 0:
        return [], [], []

    class_ids = np.argmax(scores_raw, axis=1)
    confidences = scores_raw[np.arange(len(class_ids)), class_ids]

    mask = confidences >= conf_thresh
    boxes_raw = boxes_raw[mask]
    confidences = confidences[mask]
    class_ids = class_ids[mask]

    if len(boxes_raw) == 0:
        return [], [], []

    # Scale from model input space to original frame
    boxes_xyxy = xywh2xyxy(boxes_raw)
    boxes_xyxy[:, [0, 2]] = (boxes_xyxy[:, [0, 2]] * INPUT_W - pad_x) / scale
    boxes_xyxy[:, [1, 3]] = (boxes_xyxy[:, [1, 3]] * INPUT_H - pad_y) / scale
    boxes_xyxy = np.clip(boxes_xyxy, 0,
                         [orig_w, orig_h, orig_w, orig_h])

    keep = nms(boxes_xyxy, confidences, iou_thresh)
    return boxes_xyxy[keep].astype(int), confidences[keep], class_ids[keep]


# ── Drawing ───────────────────────────────────────────────────────────────────

def draw_detections(frame, boxes, confidences, class_ids):
    for box, conf, cls in zip(boxes, confidences, class_ids):
        x1, y1, x2, y2 = box
        color = COLORS[cls].tolist()
        label = f"{COCO_CLASSES[cls]} {conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    hef_path = Path(HEF_PATH)
    if not hef_path.exists():
        raise FileNotFoundError(f"HEF not found: {hef_path}")

    hef = hailo_platform.HEF(str(hef_path))

    with hailo_platform.VDevice() as device:
        configure_params = hailo_platform.ConfigureParams.create_from_hef(
            hef, interface=hailo_platform.HailoStreamInterface.PCIe
        )
        network_groups = device.configure(hef, configure_params)
        network_group = network_groups[0]
        network_group_params = network_group.create_params()

        input_params  = hailo_platform.InputVStreamParams.make(network_group,
                            format_type=hailo_platform.FormatType.UINT8)
        output_params = hailo_platform.OutputVStreamParams.make(network_group,
                            format_type=hailo_platform.FormatType.FLOAT32)

        input_info  = hef.get_input_vstream_infos()[0]
        output_info = hef.get_output_vstream_infos()[0]
        print(f"Input  : {input_info.name}  shape={input_info.shape}")
        print(f"Output : {output_info.name}  shape={output_info.shape}")

        cap = cv2.VideoCapture(VIDEO_SOURCE)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {VIDEO_SOURCE}")

        fps_t = time.time()
        fps = 0.0
        frame_idx = 0

        with hailo_platform.InferVStreams(network_group, input_params, output_params) as infer_pipeline:
            with network_group.activate(network_group_params):
                print("Running — press Q to quit")
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        print("Cannot read frame.")
                        time.sleep(1)
                        continue
                    
                    orig_h, orig_w = frame.shape[:2]
                    blob, scale, pad_x, pad_y = preprocess(frame)
                    input_data = {input_info.name: blob[np.newaxis]}  # NHWC
         
                    raw = infer_pipeline.infer(input_data)
                    output = raw[output_info.name]
 
                    boxes, confs, cls_ids = postprocess(
                        output, scale, pad_x, pad_y, orig_w, orig_h)
                    frame_idx += 1
                    print(f"Frame {frame_idx}: detections={len(boxes)}")
                    frame = draw_detections(frame, boxes, confs, cls_ids)

                    # FPS overlay
                    now = time.time()
                    fps = 0.9 * fps + 0.1 / max(now - fps_t, 1e-6)
                    fps_t = now
                    cv2.putText(frame, f"FPS {fps:.1f}", (10, 28),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                    cv2.imshow("Hailo-8 YOLOv8n",frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
