#!/usr/bin/env python3
"""
Hailo-8 Object Detection with YOLOv8n + Picamera2 (Raspberry Pi 5)
Requires: hailo_platform, picamera2, OpenCV, numpy
"""
import threading
import cv2 as cv
import numpy as np
import time
import os
import sys
from pathlib import Path
from matplotlib import pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hailo_platform import (
    HEF, VDevice, HailoStreamInterface,
    InferVStreams, ConfigureParams,
    InputVStreamParams, OutputVStreamParams, FormatType,
)

from picamera.cameras import get_frame, camera_thread, stop_cameras
from stereovision import draw_depth


# ── Config ─────────────────────────────────────────────────────────────────────
HEF_PATH     = "./models/yolov8n.hef"
CONF_THRESH  = 0.40
IOU_THRESH   = 0.40
INPUT_W      = 640
INPUT_H      = 640
DISPLAY_W    = 1280        # preview window size
DISPLAY_H    = 720
LOG_EVERY_N  = 50
APPLY_HOST_NMS = True

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


# ── Preprocessing ──────────────────────────────────────────────────────────────

def preprocess_lores(lores_frame: np.ndarray):
    """
    lores_frame is already INPUT_W×INPUT_H RGB from the ISP.
    Keep it as uint8 — no letterbox needed because picamera2 centre-crops
    the lores stream to the requested aspect ratio.
    """
    if lores_frame.dtype != np.uint8:
        lores_frame = lores_frame.astype(np.uint8, copy=False)
    elif not lores_frame.flags["C_CONTIGUOUS"]:
        lores_frame = np.ascontiguousarray(lores_frame)
    return lores_frame, 1.0, 0, 0


def letterbox_preprocess(frame: np.ndarray):
    """Fallback: letterbox BGR frame → INPUT_W×INPUT_H RGB uint8."""
    h, w = frame.shape[:2]
    scale = min(INPUT_W / w, INPUT_H / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv.resize(frame, (nw, nh), interpolation=cv.INTER_LINEAR)
    canvas = np.full((INPUT_H, INPUT_W, 3), 114, dtype=np.uint8)
    pad_x = (INPUT_W - nw) // 2
    pad_y = (INPUT_H - nh) // 2
    canvas[pad_y:pad_y+nh, pad_x:pad_x+nw] = resized
    rgb = cv.cvtColor(canvas, cv.COLOR_BGR2RGB)
    return rgb.astype(np.uint8), scale, pad_x, pad_y


# ── Postprocessing ─────────────────────────────────────────────────────────────

def xywh2xyxy(boxes: np.ndarray) -> np.ndarray:
    b = boxes.copy()
    b[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    b[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    b[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    b[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    return b


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float):
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]; keep.append(i)
        if order.size == 1: break
        xx1 = np.maximum(boxes[i,0], boxes[order[1:],0])
        yy1 = np.maximum(boxes[i,1], boxes[order[1:],1])
        xx2 = np.minimum(boxes[i,2], boxes[order[1:],2])
        yy2 = np.minimum(boxes[i,3], boxes[order[1:],3])
        inter = np.maximum(0, xx2-xx1) * np.maximum(0, yy2-yy1)
        ai = (boxes[i,2]-boxes[i,0])*(boxes[i,3]-boxes[i,1])
        ao = (boxes[order[1:],2]-boxes[order[1:],0])*(boxes[order[1:],3]-boxes[order[1:],1])
        iou = inter / (ai + ao - inter + 1e-6)
        order = order[1:][iou <= iou_thresh]
    return keep


def postprocess(raw: np.ndarray, scale, pad_x, pad_y, orig_w, orig_h):
    """
    raw shape: (1, 84, 8400) — YOLOv8 anchor-free
    Coordinates are in model-input pixel space (0..INPUT_W / 0..INPUT_H).
    """
    if isinstance(raw, dict):
        raw = next(iter(raw.values()))
    if isinstance(raw, (list, tuple)):
        raw = raw[0] if len(raw) == 1 else raw
        if isinstance(raw, (list, tuple)):
            boxes_list = []
            confs_list = []
            cls_list = []

            for class_id, class_dets in enumerate(raw):
                class_arr = np.asarray(class_dets)
                if class_arr.size == 0:
                    continue
                class_arr = class_arr.reshape(-1, class_arr.shape[-1])

                for det in class_arr:
                    ymin, xmin, ymax, xmax, conf = det[:5]
                    conf = float(conf)
                    if conf < CONF_THRESH:
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

            if APPLY_HOST_NMS:
                keep = nms(boxes_xyxy, confidences, IOU_THRESH)
                return boxes_xyxy[keep].astype(int), confidences[keep], class_ids[keep]
            return boxes_xyxy.astype(int), confidences, class_ids

    out = np.asarray(raw[0] if isinstance(raw, np.ndarray) and raw.ndim == 3 else raw)

    if out.ndim != 2 or out.shape[0] < 5 or out.shape[1] == 0:
        return [], [], []
    boxes_raw = out[:4, :].T        # (8400, 4)  cx,cy,w,h
    scores_raw = out[4:, :].T        # (8400, 80)

    class_ids   = np.argmax(scores_raw, axis=1)
    confidences = scores_raw[np.arange(len(class_ids)), class_ids]

    mask = confidences >= CONF_THRESH
    boxes_raw, confidences, class_ids = (
        boxes_raw[mask], confidences[mask], class_ids[mask])

    if len(boxes_raw) == 0:
        return [], [], []

    boxes = xywh2xyxy(boxes_raw)
    # Map from model input space → display frame space
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / scale * (orig_w / INPUT_W)
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / scale * (orig_h / INPUT_H)
    boxes = np.clip(boxes, 0, [orig_w, orig_h, orig_w, orig_h])

    if APPLY_HOST_NMS:
        keep = nms(boxes, confidences, IOU_THRESH)
        return boxes[keep].astype(int), confidences[keep], class_ids[keep]
    return boxes.astype(int), confidences, class_ids


# ── Drawing ────────────────────────────────────────────────────────────────────

def draw(frame, boxes, confidences, class_ids):
    for box, conf, cls in zip(boxes, confidences, class_ids):
        x1, y1, x2, y2 = box
        color = COLORS[cls].tolist()
        label = f"{COCO_CLASSES[cls]} {conf:.2f}"
        cv.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv.rectangle(frame, (x1, y1-th-6), (x1+tw+4, y1), color, -1)
        cv.putText(frame, label, (x1+2, y1-4),
                    cv.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv.LINE_AA)
    return frame


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    hef = HEF(str(Path(HEF_PATH)))

    show_window = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    print(f"Display detected: {show_window}. Running in {'windowed' if show_window else 'headless'} mode.")

    t0 = threading.Thread(target=camera_thread, args=(0, "cam_0"), daemon=True)
    t1 = threading.Thread(target=camera_thread, args=(1, "cam_1"), daemon=True)
    t0.start()
    t1.start()
    time.sleep(2.0) # Give cameras a brief window to initialize and spin upq

    with VDevice() as device:
        cfg_params = ConfigureParams.create_from_hef(
            hef, interface=HailoStreamInterface.PCIe)
        ng = device.configure(hef, cfg_params)[0]
        ng_params = ng.create_params()

        in_params  = InputVStreamParams.make(ng,  format_type=FormatType.UINT8)
        out_params = OutputVStreamParams.make(ng, format_type=FormatType.FLOAT32)

        in_info  = hef.get_input_vstream_infos()[0]
        out_info = hef.get_output_vstream_infos()[0]
        print(f"Model input  : {in_info.name}  {in_info.shape}")
        print(f"Model output : {out_info.name}  {out_info.shape}")

        fps_t = time.perf_counter()
        fps   = 0.0
        frame_idx = 0

        #cv.namedWindow("Hailo-8 YOLOv8n - Pi5", cv.WINDOW_NORMAL)

        with InferVStreams(ng, in_params, out_params) as pipeline:
            with ng.activate(ng_params):
                print("Running — press Q to quit")
                while True:
                    # Grab both streams when rendering preview.
                    frame_r = get_frame("cam_0")
                    frame_l = get_frame("cam_1")
                    lores_frame = get_frame("lores_0")

                    blob, scale, px, py = preprocess_lores(lores_frame)
                    raw = pipeline.infer({in_info.name: blob[np.newaxis]})
                    output = raw[out_info.name]  # (1, 84, 8400)

                    boxes, confs, cls_ids = postprocess(
                        output, scale, px, py, DISPLAY_W, DISPLAY_H)
                    draw(frame_r, boxes, confs, cls_ids)
                    
                    # FPS
                    frame_idx += 1
                    now  = time.perf_counter()
                    fps  = 0.9 * fps + 0.1 / max(now - fps_t, 1e-6)
                    fps_t = now
                    cv.putText(frame_r, f"FPS {fps:.1f} detections={len(boxes)} ", (10, 32),
                                cv.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)

                    disparity = draw_depth(frame_r,frame_l)
                    disparity = cv.normalize(disparity, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)  
                    cv.imshow("camera right", frame_r)
                    cv.imshow("camera left", frame_l)
                    cv.moveWindow("camera right", 0, 0)
                    cv.moveWindow("camera left", 1282, 0)  
                    cv.imshow("disparity", disparity)
                    cv.moveWindow("disparity", 1280, 800)    

                    if cv.waitKey(1) & 0xFF == ord("q"):
                        stop_cameras()
                        t0.join()
                        t1.join()
                        break

    cv.destroyAllWindows()
    print("[INFO] Program terminated gracefully.")


if __name__ == "__main__":
    main()
