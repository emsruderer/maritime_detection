"""
Stereo Depth Estimation
Berechnet Tiefenkarten aus rektifizierten Stereo-Bildern.
"""

import cv2
import numpy as np
import json
from pathlib import Path
from dataclasses import dataclass


@dataclass
class DepthEstimator:
    calib_path: str

    def __post_init__(self):
        with open(self.calib_path) as f:
            calib = json.load(f)

        base_dir = Path(self.calib_path).parent
        self.map_lx = np.load(base_dir / "map_lx.npy")
        self.map_ly = np.load(base_dir / "map_ly.npy")
        self.map_rx = np.load(base_dir / "map_rx.npy")
        self.map_ry = np.load(base_dir / "map_ry.npy")
        self.Q = np.array(calib["Q"])
        self.baseline_mm = calib["baseline_mm"]

        # SGBM Stereo Matcher (robust für Außenszenen)
        self.stereo = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=128,   # muss Vielfaches von 16 sein
            blockSize=7,
            P1=8  * 3 * 7 ** 2,
            P2=32 * 3 * 7 ** 2,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=32,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )

    def rectify(self, img_left: np.ndarray, img_right: np.ndarray):
        """Rektifiziert beide Kamerabilder."""
        rect_l = cv2.remap(img_left,  self.map_lx, self.map_ly, cv2.INTER_LINEAR)
        rect_r = cv2.remap(img_right, self.map_rx, self.map_ry, cv2.INTER_LINEAR)
        return rect_l, rect_r

    def compute_depth(self, img_left: np.ndarray, img_right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Berechnet Disparitäts- und Tiefenkarte.
        
        Returns:
            depth_map: Tiefe in Metern (float32)
            disparity:  Rohe Disparitätskarte
        """
        rect_l, rect_r = self.rectify(img_left, img_right)

        gray_l = cv2.cvtColor(rect_l, cv2.COLOR_BGR2GRAY) if rect_l.ndim == 3 else rect_l
        gray_r = cv2.cvtColor(rect_r, cv2.COLOR_BGR2GRAY) if rect_r.ndim == 3 else rect_r

        disparity = self.stereo.compute(gray_l, gray_r).astype(np.float32) / 16.0

        # 3D-Punktwolke berechnen (Q-Matrix aus Kalibrierung)
        points_3d = cv2.reprojectImageTo3D(disparity, self.Q)
        depth_map = points_3d[:, :, 2] / 1000.0  # mm → Meter

        # Ungültige Werte maskieren
        depth_map[disparity <= 0] = 0.0
        depth_map[depth_map < 0]  = 0.0
        depth_map[depth_map > 200] = 0.0  # max 200m

        return depth_map, disparity, rect_l

    def get_object_depth(self, depth_map: np.ndarray, bbox: tuple) -> float:
        """
        Mittlere Tiefe eines Bounding-Box-Bereichs.
        
        Args:
            bbox: (x1, y1, x2, y2) in Pixeln
        Returns:
            Tiefe in Metern
        """
        x1, y1, x2, y2 = map(int, bbox)
        roi = depth_map[y1:y2, x1:x2]
        valid = roi[roi > 0]
        if len(valid) == 0:
            return -1.0
        # Median ist robuster als Mittelwert
        return float(np.median(valid))

    def depth_colormap(self, depth_map: np.ndarray, max_dist: float = 50.0) -> np.ndarray:
        """Erstellt eine farbige Visualisierung der Tiefenkarte."""
        norm = np.clip(depth_map / max_dist, 0, 1)
        colored = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        colored[depth_map == 0] = [50, 50, 50]  # Ungültige Pixel grau
        return colored
