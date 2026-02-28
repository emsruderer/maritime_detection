"""
Maritime Object Detection - Main Pipeline
Kombiniert Stereo-Tiefe + Hailo-8 Objekterkennung in Echtzeit.

Usage:
    python main.py --hef ./models/maritime.hef --calib ./calib_images/stereo_calibration.json
    python main.py --demo   # Demo-Modus mit synthetischen Daten
"""

import cv2
import numpy as np
import argparse
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from calibration.depth_estimator import DepthEstimator
from inference.hailo_detector   import HailoDetector


# ── Farben pro Klasse ──────────────────────────────────────────────────────────
CLASS_COLORS = {
    "buoy":            (0, 255, 255),   # Gelb
    "boat":            (0, 128, 255),   # Orange
    "obstacle":        (0, 0, 255),     # Rot
    "person_in_water": (0, 255, 0),     # Grün  ← höchste Priorität!
}


def draw_detections(frame: np.ndarray, detections: list, depth_map: np.ndarray = None) -> np.ndarray:
    """Zeichnet Bounding Boxes mit Tiefenangabe auf das Bild."""
    vis = frame.copy()

    for det in detections:
        x1, y1, x2, y2 = det.bbox
        color = CLASS_COLORS.get(det.class_name, (255, 255, 255))

        # Bounding Box
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

        # Label
        depth_str = f" {det.depth_m:.1f}m" if det.depth_m > 0 else ""
        label = f"{det.class_name} {det.confidence:.0%}{depth_str}"

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(vis, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(vis, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

    return vis


def run_pipeline(hef_path: str, calib_path: str, cam_left: int = 0, cam_right: int = 1):
    """Echtzeit-Pipeline mit zwei Kameras."""

    print("🚀 Starte Maritime Detection Pipeline...")
    print(f"   📷 Kamera Links:  /dev/video{cam_left}")
    print(f"   📷 Kamera Rechts: /dev/video{cam_right}")

    # Komponenten initialisieren
    detector = HailoDetector(hef_path, conf_threshold=0.5)
    depth_est = DepthEstimator(calib_path)

    cap_l = cv2.VideoCapture(cam_left)
    cap_r = cv2.VideoCapture(cam_right)

    # Kamera-Einstellungen
    for cap in [cap_l, cap_r]:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)
        cap.set(cv2.CAP_PROP_FPS,            30)

    fps_counter = 0
    fps_start   = time.time()
    fps_display = 0.0

    print("✅ Pipeline läuft! [Q] zum Beenden")

    while True:
        ret_l, frame_l = cap_l.read()
        ret_r, frame_r = cap_r.read()

        if not ret_l or not ret_r:
            print("❌ Kamera-Fehler!")
            break

        # ── 1. Tiefenkarte berechnen ──────────────────────────────────────────
        depth_map, disparity, rect_l = depth_est.compute_depth(frame_l, frame_r)

        # ── 2. Objekterkennung auf linkem (rektifiziertem) Bild ───────────────
        detections = detector.infer(rect_l)

        # ── 3. Tiefe pro Detektion befüllen ───────────────────────────────────
        for det in detections:
            det.depth_m = depth_est.get_object_depth(depth_map, det.bbox)

        # ── 4. Visualisierung ─────────────────────────────────────────────────
        vis_det   = draw_detections(rect_l, detections)
        vis_depth = depth_est.depth_colormap(depth_map, max_dist=100.0)

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
        cv2.imshow("Maritime Detection | Links: Erkennung | Rechts: Tiefe", combined)

        # ── 5. Warnungen ausgeben ─────────────────────────────────────────────
        for det in detections:
            if det.depth_m > 0 and det.depth_m < 20.0:
                print(f"⚠️  NAHES OBJEKT: {det} @ {det.depth_m:.1f}m")

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap_l.release()
    cap_r.release()
    cv2.destroyAllWindows()


def run_demo():
    """Demo-Modus ohne echte Kameras oder Hailo."""
    print("🎮 Demo-Modus (synthetische Daten)")

    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    # Himmel
    frame[:300] = [200, 150, 80]
    # Wasser
    frame[300:] = [120, 80, 30]

    # Simulierte Detektionen
    from inference.hailo_detector import Detection
    detections = [
        Detection(0, "buoy",     0.92, (300, 380, 360, 440), depth_m=45.2),
        Detection(1, "boat",     0.87, (700, 320, 900, 450), depth_m=120.5),
        Detection(2, "obstacle", 0.75, (100, 400, 200, 480), depth_m=18.3),
    ]

    vis = draw_detections(frame, detections)
    cv2.putText(vis, "DEMO MODE | Maritime Detection", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

    cv2.imshow("Maritime Detection Demo", vis)
    print("Drücke eine Taste zum Beenden...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Maritime Object Detection")
    parser.add_argument("--hef",   default="./models/maritime.hef",
                        help="Pfad zur Hailo .hef Datei")
    parser.add_argument("--calib", default="./calib_images/stereo_calibration.json",
                        help="Pfad zur Kalibrierungsdatei")
    parser.add_argument("--cam-left",  type=int, default=0, help="Kamera-Index links")
    parser.add_argument("--cam-right", type=int, default=1, help="Kamera-Index rechts")
    parser.add_argument("--demo", action="store_true", help="Demo ohne Hardware")
    args = parser.parse_args()

    if args.demo:
        run_demo()
    else:
        run_pipeline(args.hef, args.calib, args.cam_left, args.cam_right)
