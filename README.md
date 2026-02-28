# 🚢 Maritime Object Detection
**Bojen & Hindernisse erkennen mit Stereo-Kameras + Hailo-8 auf RPi 5**

---

## 🏗️ Architektur

```
Kamera Links ──┐
               ├──► Stereo-Rektifikation ──► Tiefenkarte (SGBM)
Kamera Rechts ─┘                                    │
                                                     ▼
               ┌──── Linkes Bild ────► Hailo-8 (YOLOv8) ──► Bounding Boxes
               │                                             │
               └─────────────────────────────────────────────┘
                                   │
                                   ▼
                        3D-Position (x, y, Tiefe in Metern)
```

---

## 🚀 Schnellstart

### Schritt 1: Installation
```bash
pip install -r requirements.txt
```

### Schritt 2: Stereo-Kalibrierung
```bash
# Schachbrettmuster ausdrucken (9x6, 25mm Felder)
# Bildpaare aufnehmen → calib_images/left_001.jpg + right_001.jpg
python calibration/stereo_calibrate.py --images ./calib_images
```

### Schritt 3: Modell exportieren (auf x86 PC)
```bash
# Standard YOLOv8 oder eigenes trainiertes Modell
python models/export_model.py --model yolov8n.pt --output ./models --verify
```

### Schritt 4: Hailo Kompilierung (auf x86 PC mit Hailo DFC)
```bash
hailomz compile --model ./models/yolov8n.onnx --hw-arch hailo8
# → erzeugt ./models/yolov8n.hef
```

### Schritt 5: Auf RPi 5 ausführen
```bash
python main.py --hef ./models/yolov8n.hef --calib ./calib_images/stereo_calibration.json
```

### Demo (ohne Hardware)
```bash
python main.py --demo
```

---

## 📁 Projektstruktur

```
maritime_detection/
├── main.py                          # Haupt-Pipeline
├── requirements.txt
├── calibration/
│   ├── stereo_calibrate.py          # Kamera-Kalibrierung
│   └── depth_estimator.py           # Tiefenkarten-Berechnung
├── models/
│   └── export_model.py              # PyTorch → ONNX Export
├── inference/
│   └── hailo_detector.py            # Hailo-8 Inferenz Engine
└── data/
    ├── raw/                         # Rohe Kamerabilder
    └── processed/                   # Verarbeitete Daten
```

---

## ⚙️ Kamera-Setup Empfehlungen

| Parameter    | Empfehlung                        |
|-------------|-----------------------------------|
| Baseline    | 10–20 cm (für 5–150m Reichweite)  |
| Auflösung   | 1280×720 @ 30fps                  |
| Kameratyp   | RPi Camera Module 3 oder IMX477   |
| Ausrichtung | Horizontal, parallel              |

---

## 🎯 Eigenes Modell trainieren

Für maritime Objekte empfiehlt sich:
- **Datensatz**: [SeaDronesSee](https://github.com/Ben93kie/SeaDronesSee) oder eigene Bilder
- **Vortrainiert**: YOLOv8n (schnell) oder YOLOv8s (genauer)
- **Augmentierung**: Starke Helligkeit/Kontrast-Variationen (Sonnenlicht auf Wasser!)

```bash
yolo train model=yolov8n.pt data=maritime.yaml epochs=100 imgsz=640
```
