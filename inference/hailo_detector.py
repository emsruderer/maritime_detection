"""
Hailo-8 Inference Engine
Lädt ein .hef Modell und führt Objekterkennung durch.
Kompatibel mit Hailo Python SDK (hailo_platform).
"""

import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Detection:
    class_id:   int
    class_name: str
    confidence: float
    bbox:       tuple   # (x1, y1, x2, y2) in Pixeln
    depth_m:    float = -1.0  # wird später aus Tiefenkarte befüllt

    @property
    def center(self):
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    def __str__(self):
        depth_str = f" | {self.depth_m:.1f}m" if self.depth_m > 0 else ""
        return f"{self.class_name} ({self.confidence:.0%}){depth_str}"


MARITIME_CLASSES = {
    0: "buoy",
    1: "boat",
    2: "obstacle",
    3: "person_in_water",
}


class HailoDetector:
    """
    Objekterkennung mit Hailo-8 NPU.
    Benötigt hailo_platform SDK (auf RPi 5 mit Hailo-8 installiert).
    """

    def __init__(self, hef_path: str, input_size: tuple = (640, 640), conf_threshold: float = 0.5):
        self.hef_path = hef_path
        self.input_size = input_size
        self.conf_threshold = conf_threshold
        self._init_hailo()

    def _init_hailo(self):
        """Initialisiert Hailo-8 Hardware."""
        try:
            from hailo_platform import (
                HEF, VDevice, HailoStreamInterface,
                InferVStreams, ConfigureParams, InputVStreamParams, OutputVStreamParams
            )

            self.hef    = HEF(self.hef_path)
            self.device = VDevice()

            configure_params = ConfigureParams.create_from_hef(
                self.hef, interface=HailoStreamInterface.PCIe
            )
            self.network_groups = self.device.configure(self.hef, configure_params)
            self.network_group  = self.network_groups[0]
            self.network_group_params = self.network_group.create_params()

            self.input_vstream_params  = InputVStreamParams.make(self.network_group)
            self.output_vstream_params = OutputVStreamParams.make(self.network_group)

            print(f"✅ Hailo-8 initialisiert: {self.hef_path}")
            self._use_hailo = True

        except ImportError:
            print("⚠️  hailo_platform nicht verfügbar – Fallback auf ONNX Runtime")
            self._init_onnx_fallback()
            self._use_hailo = False

    def _init_onnx_fallback(self):
        """ONNX Runtime Fallback (für Entwicklung ohne Hailo)."""
        import onnxruntime as ort
        onnx_path = self.hef_path.replace(".hef", ".onnx")
        self.ort_session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self.input_name  = self.ort_session.get_inputs()[0].name
        print(f"✅ ONNX Fallback geladen: {onnx_path}")

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Bereitet Bild für Inferenz vor (resize + normalize)."""
        resized = cv2.resize(frame, self.input_size)
        rgb     = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor  = rgb.astype(np.float32) / 255.0
        return tensor[np.newaxis, ...]  # (1, H, W, 3)

    def postprocess(self, raw_output: np.ndarray, orig_shape: tuple) -> list[Detection]:
        """
        Verarbeitet Roh-Output des YOLO-Modells.
        raw_output shape: (1, num_detections, 4+num_classes)
        """
        detections = []
        orig_h, orig_w = orig_shape[:2]
        scale_x = orig_w / self.input_size[0]
        scale_y = orig_h / self.input_size[1]

        predictions = raw_output[0]  # (num_det, 4+classes)

        for pred in predictions:
            cx, cy, w, h = pred[:4]
            class_scores  = pred[4:]
            class_id      = int(np.argmax(class_scores))
            confidence     = float(class_scores[class_id])

            if confidence < self.conf_threshold:
                continue

            # Center-Format → Corner-Format, skaliert auf Originalgröße
            x1 = int((cx - w / 2) * scale_x)
            y1 = int((cy - h / 2) * scale_y)
            x2 = int((cx + w / 2) * scale_x)
            y2 = int((cy + h / 2) * scale_y)

            # Clipping
            x1, x2 = max(0, x1), min(orig_w, x2)
            y1, y2 = max(0, y1), min(orig_h, y2)

            detections.append(Detection(
                class_id=class_id,
                class_name=MARITIME_CLASSES.get(class_id, f"class_{class_id}"),
                confidence=confidence,
                bbox=(x1, y1, x2, y2),
            ))

        return detections

    def infer(self, frame: np.ndarray) -> list[Detection]:
        """Vollständige Inferenz: Vorverarbeitung → NPU → Nachverarbeitung."""
        tensor = self.preprocess(frame)

        if self._use_hailo:
            raw = self._hailo_infer(tensor)
        else:
            raw = self._onnx_infer(tensor)

        return self.postprocess(raw, frame.shape)

    def _hailo_infer(self, tensor: np.ndarray) -> np.ndarray:
        from hailo_platform import InferVStreams
        with InferVStreams(
            self.network_group,
            self.input_vstream_params,
            self.output_vstream_params
        ) as infer_pipeline:
            with self.network_group.activate(self.network_group_params):
                input_data = {
                    self.hef.get_input_vstream_infos()[0].name: tensor
                }
                raw_detections = infer_pipeline.infer(input_data)
                return list(raw_detections.values())[0]

    def _onnx_infer(self, tensor: np.ndarray) -> np.ndarray:
        # ONNX erwartet (1, 3, H, W) statt (1, H, W, 3)
        tensor_chw = np.transpose(tensor, (0, 3, 1, 2))
        return self.ort_session.run(None, {self.input_name: tensor_chw})[0]
