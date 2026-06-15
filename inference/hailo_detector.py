"""
Hailo-8 Inference Engine
Lädt ein .hef Modell und führt Objekterkennung durch.
Kompatibel mit Hailo Python SDK (hailo_platform).
"""

import numpy as np
import cv2
from dataclasses import dataclass
from pathlib import Path


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
    """Objekterkennung mit Hailo-8 NPU."""

    def __init__(self, model_path: str, input_size: tuple = (640, 640), conf_threshold: float = 0.5):
        self.model_path = model_path
        self.hef_path = model_path  # backward compat
        self.input_size = input_size
        self.conf_threshold = conf_threshold
        self._backend = None
        self._init_hailo()

    def _init_hailo(self):
        """Initialisiert Hailo-8 Hardware."""
        try:
            from hailo_platform import (
                HEF, VDevice, HailoStreamInterface,
                InferVStreams, ConfigureParams, InputVStreamParams, OutputVStreamParams,
                FormatType,
            )

            self.hef    = HEF(self.model_path)
            self.device = VDevice()

            configure_params = ConfigureParams.create_from_hef(
                self.hef, interface=HailoStreamInterface.PCIe
            )
            self.network_groups = self.device.configure(self.hef, configure_params)
            self.network_group  = self.network_groups[0]
            self.network_group_params = self.network_group.create_params()

            # UINT8 matches the uint8 RGB frames from preprocess() — no per-frame conversion
            self.input_vstream_params  = InputVStreamParams.make(
                self.network_group, format_type=FormatType.UINT8)
            self.output_vstream_params = OutputVStreamParams.make(
                self.network_group, format_type=FormatType.FLOAT32)

            # Build persistent InferVStreams context to avoid per-frame setup overhead
            self._infer_pipeline = InferVStreams(
                self.network_group,
                self.input_vstream_params,
                self.output_vstream_params,
            )
            self._infer_pipeline.__enter__()
            self._network_activation = self.network_group.activate(self.network_group_params)
            self._network_activation.__enter__()

            self._in_name  = self.hef.get_input_vstream_infos()[0].name
            self._out_name = self.hef.get_output_vstream_infos()[0].name

            print(f"✅ Hailo-8 initialisiert: {self.model_path}")
            self._backend = "hailo"

        except Exception as exc:
            raise RuntimeError(f"Hailo-8 Initialisierung fehlgeschlagen: {exc}") from exc

    def preprocess(self, lores_frame: np.ndarray) -> np.ndarray:
        """
        lores_frame is already INPUT_W×INPUT_H RGB from the ISP.
        Returns uint8 array ready for Hailo UINT8 input stream.
        """
        if lores_frame.dtype != np.uint8:
            lores_frame = lores_frame.astype(np.uint8, copy=False)
        elif not lores_frame.flags["C_CONTIGUOUS"]:
            lores_frame = np.ascontiguousarray(lores_frame)
        return lores_frame


    def postprocess(self, raw_output, orig_shape: tuple) -> list[Detection]:
        """
        Handles Hailo NMS nested list output: list of 80 per-class arrays
        each shaped (N, 5) with [ymin, xmin, ymax, xmax, conf] in normalised coords.
        Also handles raw anchor tensor (1, 84, 8400) for ONNX fallback.
        """
        orig_h, orig_w = orig_shape[:2]
        detections = []

        # ── Hailo NMS path: nested list ──────────────────────────────────────
        if isinstance(raw_output, (list, tuple)):
            raw_output = raw_output[0] if len(raw_output) == 1 else raw_output
            if isinstance(raw_output, (list, tuple)):
                for class_id, class_dets in enumerate(raw_output):
                    class_arr = np.asarray(class_dets)
                    if class_arr.size == 0:
                        continue
                    class_arr = class_arr.reshape(-1, class_arr.shape[-1])
                    for det in class_arr:
                        ymin, xmin, ymax, xmax, conf = det[:5]
                        conf = float(conf)
                        if conf < self.conf_threshold:
                            continue
                        if max(abs(xmin), abs(ymin), abs(xmax), abs(ymax)) <= 1.5:
                            x1 = int(float(xmin) * orig_w)
                            y1 = int(float(ymin) * orig_h)
                            x2 = int(float(xmax) * orig_w)
                            y2 = int(float(ymax) * orig_h)
                        else:
                            x1, y1, x2, y2 = int(xmin), int(ymin), int(xmax), int(ymax)
                        x1, x2 = max(0, x1), min(orig_w, x2)
                        y1, y2 = max(0, y1), min(orig_h, y2)
                        if x2 <= x1 or y2 <= y1:
                            continue
                        detections.append(Detection(
                            class_id=class_id,
                            class_name=MARITIME_CLASSES.get(class_id, f"class_{class_id}"),
                            confidence=conf,
                            bbox=(x1, y1, x2, y2),
                        ))
                return detections
            raw_output = np.asarray(raw_output)

        # ── Raw anchor tensor path: (1, 84, 8400) ────────────────────────────
        raw_output = np.asarray(raw_output)
        out = raw_output[0] if raw_output.ndim == 3 else raw_output
        if out.ndim != 2 or out.shape[0] < 5:
            return detections

        boxes_raw  = out[:4, :].T
        scores_raw = out[4:, :].T
        class_ids  = np.argmax(scores_raw, axis=1)
        confs      = scores_raw[np.arange(len(class_ids)), class_ids]
        mask       = confs >= self.conf_threshold
        boxes_raw, confs, class_ids = boxes_raw[mask], confs[mask], class_ids[mask]

        scale_x = orig_w / self.input_size[0]
        scale_y = orig_h / self.input_size[1]
        for box, conf, cid in zip(boxes_raw, confs, class_ids):
            cx, cy, w, h = box
            x1 = max(0, int((cx - w / 2) * scale_x))
            y1 = max(0, int((cy - h / 2) * scale_y))
            x2 = min(orig_w, int((cx + w / 2) * scale_x))
            y2 = min(orig_h, int((cy + h / 2) * scale_y))
            detections.append(Detection(
                class_id=int(cid),
                class_name=MARITIME_CLASSES.get(int(cid), f"class_{cid}"),
                confidence=float(conf),
                bbox=(x1, y1, x2, y2),
            ))
        return detections

    def infer(self, frame: np.ndarray) -> list[Detection]:
        """Vollständige Inferenz: Vorverarbeitung → Hailo-8 → Nachverarbeitung."""
        tensor = self.preprocess(frame)
        raw = self._hailo_infer(tensor)
        return self.postprocess(raw, frame.shape)

    def _hailo_infer(self, tensor: np.ndarray):
        """Run one frame through the persistent Hailo InferVStreams context."""
        raw = self._infer_pipeline.infer({self._in_name: tensor[np.newaxis]})
        return raw[self._out_name]

    def __del__(self):
        """Clean up persistent Hailo contexts on destruction."""
        try:
            self._network_activation.__exit__(None, None, None)
        except Exception:
            pass
        try:
            self._infer_pipeline.__exit__(None, None, None)
        except Exception:
            pass
        try:
            self.device.release()
        except Exception:
            pass

    