"""
Stereo Depth Estimation
Berechnet Tiefenkarten aus rektifizierten Stereo-Bildern.
"""

import cv2
import numpy as np
import json
from pathlib import Path
from dataclasses import dataclass, field
from collections import deque


@dataclass
class DepthEstimator:
    calib_path: str
    conf_threshold: float = 0.3  # Skip depth for low-confidence detections
    temporal_window: int = 3      # Rolling average over N frames
    center_region: float = 0.8    # Use 80% center of bbox (ignore noisy edges)
    depth_history: dict = field(default_factory=dict)  # Temporal buffer per object

    def __post_init__(self):
        with open(self.calib_path) as f:
            calib = json.load(f)
            print(calib)

        base_dir = Path(self.calib_path).parent
        self.map_lx = np.load(base_dir / "map_lx.npy")
        self.map_ly = np.load(base_dir / "map_ly.npy")
        self.map_rx = np.load(base_dir / "map_rx.npy")
        self.map_ry = np.load(base_dir / "map_ry.npy")
        self.Q = np.array(calib["Q"])
        self.baseline_mm = calib["baseline_mm"]
        print(f"Loaded calibration: baseline={self.baseline_mm}mm")
        print(f"Q-Matrix:\n{self.Q}")
        print(f"Rectification maps shapes: {self.map_lx.shape}, {self.map_ly.shape}, {self.map_rx.shape}, {self.map_ry.shape}")

        # SGBM Stereo Matcher – half-resolution for speed, MODE_HH for quality
        # Half-res maps are computed on demand in rectify_half()
        self.stereo = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=64,    # halved to match half-res input
            blockSize=5,
            P1=8  * 3 * 5 ** 2,
            P2=32 * 3 * 5 ** 2,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=80,
            speckleRange=16,
            mode=cv2.STEREO_SGBM_MODE_HH,
        )

    def rectify(self, img_left: np.ndarray, img_right: np.ndarray):
        """Rektifiziert beide Kamerabilder."""
        rect_l = cv2.remap(img_left,  self.map_lx, self.map_ly, cv2.INTER_LINEAR)
        rect_r = cv2.remap(img_right, self.map_rx, self.map_ry, cv2.INTER_LINEAR)
        return rect_l, rect_r

    def compute_depth(self, img_left: np.ndarray, img_right: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Berechnet Disparitäts- und Tiefenkarte.
        Läuft intern auf halber Auflösung für ~4× Speedup.

        Returns:
            depth_map: Tiefe in Metern (float32, Originalauflösung)
            disparity:  Rohe Disparitätskarte (halbe Auflösung)
            rect_l:     Rektifiziertes linkes Bild (Originalauflösung)
        """
        rect_l, rect_r = (img_left, img_right)

        # Downscale to half resolution before SGBM
        # h, w = rect_l.shape[:2]
        # half_l = cv2.resize(rect_l, (w // 2, h // 2), interpolation=cv2.INTER_LINEAR)
        # half_r = cv2.resize(rect_r, (w // 2, h // 2), interpolation=cv2.INTER_LINEAR)

        gray_l = cv2.cvtColor(rect_l, cv2.COLOR_BGR2GRAY) 
        gray_r = cv2.cvtColor(rect_r, cv2.COLOR_BGR2GRAY) 

        #self.stereo = cv2.StereoSGBM.create(numDisparities=16, blockSize=7)
        #disparity = self.stereo.compute(gray_l,gray_r)
        self.stereo.setSpeckleRange(32)
        self.stereo.setUniquenessRatio(15)

        disparity = self.stereo.compute(gray_l, gray_r).astype(np.float32) / 16.0
        # print(f"Disparity stats: min={disparity.min():.2f}, max={disparity.max():.2f}, valid pixels={np.sum(disparity>0)}")
        
        # 3D-Punktwolke berechnen (Q-Matrix aus Kalibrierung)
        points_3d = cv2.reprojectImageTo3D(disparity, self.Q)
        depth_map = points_3d[:, :, 2] / 100000.0  # mm → Meter
        print(f"Depth map stats: min={depth_map[depth_map>0].min():.2f}m, max={depth_map.max():.2f}m, valid pixels={np.sum(depth_map>0)}")
        # Ungültige Werte maskieren
        depth_map[disparity <= 0] = 0.0
        depth_map[depth_map < 0]   = 0.0
        #depth_map[depth_map > 200] = 0.0  # max 200m
        #print(f"Depth map stats: min={depth_map[depth_map>0].min():.2f}m, max={depth_map.max():.2f}m, valid pixels={np.sum(depth_map>0)}")
        return depth_map, disparity, rect_l

    def get_object_depth(self, depth_map: np.ndarray, bbox: tuple, confidence: float = 1.0, object_id: str = None) -> float:
        """
        Robuste Tiefenschätzung mit Vertrauensfilterung und zeitlicher Glättung.
        
        Args:
            depth_map: Tiefenkarte (float32, Meter)
            bbox: (x1, y1, x2, y2) in Pixeln
            confidence: Detektionsvertrauen [0, 1]
            object_id: Eindeutige ID für zeitliche Glättung (z.B. Track-ID)
        Returns:
            Gefilterte Tiefe in Metern, oder -1.0 wenn ungültig
        """
        # 1. Vertrauensfilter: Ignoriere Detektionen mit niedriger Konfidenz
        if confidence < self.conf_threshold:
            return -1.0
        
        x1, y1, x2, y2 = map(int, bbox)
        w, h = x2 - x1, y2 - y1
        
        # 2. Regionenverfeinerung: Nutze 80% der Bbox-Mitte (ignoriere laute Ränder)
        margin_w = int(w * (1 - self.center_region) / 2)
        margin_h = int(h * (1 - self.center_region) / 2)
        
        x1_c = x1 + margin_w
        x2_c = x2 - margin_w
        y1_c = y1 + margin_h
        y2_c = y2 - margin_h
        
        roi = depth_map[y1_c:y2_c, x1_c:x2_c]
        valid = roi[roi > 0]
        
        if len(valid) == 0:
            return -1.0
        
        # Median ist robust gegen Ausreißer
        depth_current = float(np.median(valid))
        
        # 3. Zeitliche Glättung: Rollender Durchschnitt über N Frames
        if object_id is not None:
            if object_id not in self.depth_history:
                self.depth_history[object_id] = deque(maxlen=self.temporal_window)
            
            self.depth_history[object_id].append(depth_current)
            depth_smoothed = float(np.mean(list(self.depth_history[object_id])))
            return depth_smoothed
        
        return depth_current

    def depth_colormap(self, depth_map: np.ndarray, max_dist: float = 50.0) -> np.ndarray:
        """Erstellt eine farbige Visualisierung der Tiefenkarte."""
        norm = np.clip(depth_map / max_dist, 0, 1)
        colored = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        colored[depth_map == 0] = [50, 50, 50]  # Ungültige Pixel grau
        return colored
    
    def cleanup_object(self, object_id: str):
        """Entfernt zeitlichen Puffer eines Objekts (z.B. wenn Track endet)."""
        if object_id in self.depth_history:
            del self.depth_history[object_id]
