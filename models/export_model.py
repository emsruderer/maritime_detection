"""
YOLOv8 Model Export Pipeline
PyTorch → ONNX → (Hailo Dataflow Compiler)

Usage:
    python export_model.py --model yolov8n.pt --output ./models
    python export_model.py --model ./models/maritime_custom.pt --imgsz 640
"""

import argparse
from pathlib import Path


def export_to_onnx(model_path: str, output_dir: str, imgsz: int = 640, opset: int = 11):
    """
    Exportiert YOLOv8 Modell nach ONNX (Hailo-kompatibel).
    
    Hailo-Anforderungen:
    - Opset 11 oder 13
    - Statische Input-Shape
    - Kein Dynamic Axes
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("❌ ultralytics nicht installiert: pip install ultralytics")
        return

    print(f"📦 Lade Modell: {model_path}")
    model = YOLO(model_path)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"🔄 Exportiere nach ONNX (imgsz={imgsz}, opset={opset})...")
    
    # Export mit Hailo-kompatiblen Einstellungen
    export_path = model.export(
        format="onnx",
        imgsz=imgsz,
        opset=opset,
        simplify=True,      # ONNX Simplifier für Hailo
        dynamic=False,      # Statische Shape - WICHTIG für Hailo!
    )

    # In output_dir verschieben
    import shutil
    dest = output_path / Path(export_path).name
    shutil.move(export_path, dest)

    print(f"✅ ONNX gespeichert: {dest}")
    print(f"\n📋 Nächste Schritte für Hailo:")
    print(f"   1. Hailo Dataflow Compiler installieren (auf x86 PC)")
    print(f"   2. hailo_model_zoo parse --hw-arch hailo8 {dest.name}")
    print(f"   3. Kalibrierungsbilder für Quantisierung vorbereiten")
    print(f"   4. hailo_model_zoo optimize (INT8 Quantisierung)")
    print(f"   5. hailo_model_zoo compile → .hef Datei")
    print(f"\n   Oder mit DFC direkt:")
    print(f"   hailomz compile --model {dest.name} --hw-arch hailo8")

    return str(dest)


def verify_onnx(onnx_path: str):
    """Überprüft das exportierte ONNX-Modell."""
    try:
        import onnx
        import onnxruntime as ort
        import numpy as np

        model = onnx.load(onnx_path)
        onnx.checker.check_model(model)

        # Input/Output Info
        session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        inp = session.get_inputs()[0]
        out = session.get_outputs()[0]

        print(f"\n🔍 ONNX Modell-Info:")
        print(f"   Input:  {inp.name} → shape={inp.shape}, dtype={inp.type}")
        print(f"   Output: {out.name} → shape={out.shape}")

        # Test-Inferenz
        dummy = np.random.randn(*[d if isinstance(d, int) else 1 for d in inp.shape]).astype(np.float32)
        result = session.run(None, {inp.name: dummy})
        print(f"   ✅ Test-Inferenz erfolgreich! Output shape: {result[0].shape}")

    except ImportError as e:
        print(f"⚠️  Verify übersprungen (fehlende Pakete): {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8 → ONNX Export")
    parser.add_argument("--model",  default="yolov8n.pt",  help="Pfad zum .pt Modell")
    parser.add_argument("--output", default="./models",    help="Output-Ordner")
    parser.add_argument("--imgsz",  type=int, default=640, help="Input-Bildgröße")
    parser.add_argument("--opset",  type=int, default=11,  help="ONNX Opset Version")
    parser.add_argument("--verify", action="store_true",   help="ONNX nach Export prüfen")
    args = parser.parse_args()

    onnx_path = export_to_onnx(args.model, args.output, args.imgsz, args.opset)

    if onnx_path and args.verify:
        verify_onnx(onnx_path)
