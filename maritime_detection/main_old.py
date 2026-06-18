"""
Maritime Object Detection - Main Pipeline
Kombiniert Stereo-Tiefe + Hailo-8 Objekterkennung in Echtzeit.

Usage:
    python main.py --model ./models/maritime.hef --calib ./calib_images/stereo_calibration.json
    python main.py --model yolov8n.pt            # ultralytics COCO fallback
    python main.py --demo                        # Demo-Modus mit synthetischen Daten
"""
import threading
import time
import cv2
import os
from picamera2 import Picamera2
import numpy as np
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from calibration.depth_estimator import DepthEstimator

from hailo_platform import (
    HEF, VDevice, HailoStreamInterface,
    InferVStreams, ConfigureParams,
    InputVStreamParams, OutputVStreamParams, FormatType,
)

from inference.hailo_detector   import HailoDetector

# ── Config ─────────────────────────────────────────────────────────────────────
HEF_PATH     = "./models/yolov8n.hef"
CONF_THRESH  = 0.45
IOU_THRESH   = 0.45
INPUT_W      = 640
INPUT_H      = 640
DISPLAY_W    = 1280        # preview window size
DISPLAY_H    = 720
LOG_EVERY_N  = 50
APPLY_HOST_NMS = False

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


# ── Farben pro Klasse ──────────────────────────────────────────────────────────
CLASS_COLORS = {
    "buoy":            (0, 255, 255),   # Gelb
    "boat":            (0, 128, 255),   # Orange
    "obstacle":        (0, 0, 255),     # Rot
    "person_in_water": (0, 255, 0),     # Grün  ← höchste Priorität!
}

frames = {"cam0": None, "lores_0": None, "cam1": None, "lores_1": None}
frames_lock = threading.Lock()
running = True

# ── Picamera2 setup ────────────────────────────────────────────────────────────

def make_camera(camera: int, show_window: bool) -> Picamera2:
    """
    Camera config:
      window mode  → main + lores streams (display + inference)
      headless     → lores-only stream (inference only, less ISP load)
    """
    cam = Picamera2(camera_num=camera)
    if show_window:
        config = cam.create_preview_configuration(
            main={"size": (DISPLAY_W, DISPLAY_H), "format": "BGR888"},
            lores={"size": (INPUT_W, INPUT_H), "format": "RGB888"},
            buffer_count=3,
            queue=False,
        )
    else:
        config = cam.create_preview_configuration(
            main={"size": (INPUT_W, INPUT_H), "format": "RGB888"},
            buffer_count=2,
            queue=False,
        )
    cam.configure(config)
    cam.start()
    # Let AGC/AWB settle
    time.sleep(1.0)
    return cam


def camera_thread(camera_id, frame_key, show_window: bool):
    """Handles the libcamera feed for an individual ribbon port."""
    global running, frames
    lores_key = f"lores_{frame_key[-1]}"
    picam = None
    try:
        picam = make_camera(camera_id, show_window=show_window)
        print(f"[INFO] Camera {camera_id} started successfully.")

        while running:
            if show_window:
                # Window mode: main is for visualization, lores is RGB for inference.
                arrays, _ = picam.capture_arrays(["main", "lores"])
                with frames_lock:
                    frames[frame_key] = arrays[0]
                    frames[lores_key] = arrays[1]
            else:
                # Headless mode: only main stream (640x640 RGB) exists.
                arrays, _ = picam.capture_arrays(["main"])
                frame = arrays[0]
                with frames_lock:
                    frames[frame_key] = frame
                    frames[lores_key] = frame

    except Exception as e:
        print(f"[ERROR] Camera {camera_id} failed: {e}")
    finally:
        if picam is not None:
            picam.stop()

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
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(frame, (x1, y1-th-6), (x1+tw+4, y1), color, -1)
        cv2.putText(frame, label, (x1+2, y1-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
    return frame


def draw_detections(frame: np.ndarray, detections: list, depth_map: np.ndarray = None) -> np.ndarray:
    """Zeichnet Bounding Boxes mit Tiefenangabe auf das Bild."""
    vis = frame.copy()

    for det in detections:
        x1, y1, x2, y2 = det.bbox
        color = CLASS_COLORS.get(det.class_name, (255, 255, 255))
        #color = COCO_CLASSES.get(det.class_name, (255, 255, 255))

        # Bounding Box
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

        # Label
        depth_str = f" {det.depth_m:.1f}m" if det.depth_m > 0 else ""
        label = f"{det.class_name} {det.confidence:.0%}{depth_str}"
        #label = f"{COCO_CLASSES[cls]} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(vis, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(vis, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

    return vis


def run_pipeline(
    model_path: str,
    calib_path: str,
    cam_left: int = 1,
    cam_right: int = 0,
    show_window: bool = True,
    window_debug: bool = False,
):
    """Echtzeit-Pipeline mit zwei Kameras."""
    hef = HEF(str(Path(HEF_PATH)))
    with VDevice() as device:
        cfg_params = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
        ng = device.configure(hef, cfg_params)[0]
        ng_params = ng.create_params()

        in_params  = InputVStreamParams.make(ng,  format_type=FormatType.UINT8)
        out_params = OutputVStreamParams.make(ng, format_type=FormatType.FLOAT32)

        in_info  = hef.get_input_vstream_infos()[0]
        out_info = hef.get_output_vstream_infos()[0]
        print(f"Model input  : {in_info.name}  {in_info.shape}")
        print(f"Model output : {out_info.name}  {out_info.shape}")

        print("🚀 Starte Maritime Detection Pipeline...")
        print(f"   📷 Kamera Links:  camm1")
        print(f"   📷 Kamera Rechts: camm0")

        # Komponenten initialisieren
        detector = HailoDetector(model_path, conf_threshold=0.45)
        depth_est = DepthEstimator(calib_path)

        fps_counter = 0
        fps_start   = time.time()
        fps_display = 0.0
        frame_idx   = 0

        window_name = "Maritime Detection | Links: Erkennung | Rechts: Tiefe"

        # Detect GUI availability
        show_window = show_window and bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
        if window_debug:
            ui_framework = cv2.currentUIFramework() if hasattr(cv2, "currentUIFramework") else "unknown"
            print("[WINDOW_DEBUG] DISPLAY=", os.environ.get("DISPLAY"),
                "WAYLAND_DISPLAY=", os.environ.get("WAYLAND_DISPLAY"),
                "XDG_SESSION_TYPE=", os.environ.get("XDG_SESSION_TYPE"),
                "ui=", ui_framework,
                "opencv=", cv2.__version__)
            print(f"[WINDOW_DEBUG] Requested show_window={show_window}")

        if show_window:
            try:
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(window_name, 1280, 720)
                cv2.moveWindow(window_name, 20, 20)
                if window_debug:
                    print(f"[WINDOW_DEBUG] namedWindow ok: '{window_name}'")
            except cv2.error as e:
                if window_debug:
                    print(f"[WINDOW_DEBUG] namedWindow failed: {e}")
                show_window = False
        if not show_window:
            print("⚠️  Kein GUI-Display erkannt. Läuft im Headless-Modus; drücke Ctrl+C zum Beenden.")

        print("✅ Pipeline läuft! [Q] zum Beenden" if show_window else "✅ Pipeline läuft im Headless-Modus!")

        while True:
            with frames_lock:
                frame_l = frames["cam1"]
                frame_r = frames["cam0"]

            if frame_l is None or frame_r is None:
                time.sleep(0.01)
                continue
            # ── 1. Objekterkennung zuerst (günstig, ~10ms) ───────────────────────
            with frames_lock:
                lores_l = frames["lores_1"]
            if lores_l is None:
                time.sleep(0.01)
                continue
            with InferVStreams(ng, in_params, out_params) as pipeline:
                with ng.activate(ng_params):
                    detections = detector.infer(lores_l)
                    print(detections)
                    rect_l = frame_l  # Default: keine Rektilinearisation, direkt die Erkennungsergebnisse verwenden
                    # ── 2. Tiefenkarte nur wenn nötig (teuer, ~30–60ms) ──────────────────
                    depth_map = None
                    if detections:
                        depth_map, disparity, rect_l = depth_est.compute_depth(frame_l, frame_r)
                        for det in detections:
                            # Robust depth with confidence filtering + region refinement
                            det.depth_m = depth_est.get_object_depth(
                                depth_map, 
                                det.bbox, 
                                confidence=det.confidence
                            )
                            print(det.depth_m)
                    else:
                        rect_l = frame_l
                    
                    rect_l=frame_l
                    rect_l=cv2.resize(rect_l,(DISPLAY_W, DISPLAY_H))

                    raw = pipeline.infer({in_info.name: blob[np.newaxis]})
                    output = raw[out_info.name]  # (1, 84, 8400)

                    # ── 3. Visualisierung ─────────────────────────────────────────────────
                    #vis_det   = draw_detections(rect_l, detections)
                    boxes, confs, cls_ids = postprocess(
                                    output, scale, px, py, DISPLAY_W, DISPLAY_H)
                    if show_window and rect_l is not None:
                        draw(rect_l, boxes, confs, cls_ids)

                    vis_det   = rect_l
                    vis_depth = depth_est.depth_colormap(depth_map, max_dist=100.0) if depth_map is not None else np.full_like(rect_l, 50)

            # FPS berechnen
            fps_counter += 1
            if fps_counter >= 30:
                fps_display = fps_counter / (time.time() - fps_start)
                fps_counter = 0
                fps_start   = time.time()

            cv2.putText(vis_det, f"FPS: {fps_display:.1f} | Objekte: {len(detections)}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # Beide Views nebeneinander
            combined = np.hstack([
                cv2.resize(vis_det,   (960, 540)),
                cv2.resize(vis_depth, (960, 540))
            ])
            
            frame_idx += 1
            if frame_idx % 30 == 0:
                print(f"Frame {frame_idx}: detections={len(detections)} fps={fps_display:.1f}")

            if show_window:
                cv2.imshow(window_name, combined)
                #cv2.imshow(window_name,visdet)
                if window_debug and frame_idx == 1:
                    try:
                        x, y, w, h = cv2.getWindowImageRect(window_name)
                        print(f"[WINDOW_DEBUG] First imshow ok, rect=({x},{y},{w},{h}), frame_shape={combined.shape}")
                    except cv2.error as e:
                        print(f"[WINDOW_DEBUG] getWindowImageRect failed: {e}")
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    global running
    running = False
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Maritime Object Detection")
    parser.add_argument("--model", default="./models/yolov8n.hef")
    parser.add_argument("--calib", default="./calibration/calib_images/stereo_calibration.json",
                        help="Pfad zur Kalibrierungsdatei")
    parser.add_argument("--cam-left",  type=int, default=1, help="Kamera-Index links")
    parser.add_argument("--cam-right", type=int, default=0, help="Kamera-Index rechts")
    parser.add_argument("--headless", action="store_true", help="Disable OpenCV window output")
    parser.add_argument("--window-debug", action="store_true", help="Print OpenCV window diagnostics")
    args = parser.parse_args()

    show_window = not args.headless

    # 2. Launch concurrent threads for both ribbon connectors
    t0 = threading.Thread(target=camera_thread, args=(args.cam_right, "cam0", show_window), daemon=True)
    t1 = threading.Thread(target=camera_thread, args=(args.cam_left, "cam1", show_window), daemon=True)
    t0.start()
    t1.start()
    time.sleep(2.0) # Give cameras a brief window to initialize and spin up


    model_path = args.model
    run_pipeline(
        model_path,
        args.calib,
        args.cam_left,
        args.cam_right,
        show_window=show_window,
        window_debug=args.window_debug,
    )
