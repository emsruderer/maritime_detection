"""
Stereo Camera Calibration
Kalibriert zwei Kameras mit einem Schachbrettmuster.
Usage: python stereo_calibrate.py --images ./calib_images --cols 9 --rows 6
"""

import cv2
import numpy as np
import glob
import argparse
import json
from pathlib import Path


def calibrate_stereo(image_dir: str, cols: int = 9, rows: int = 6, square_size_mm: float = 25.0):
    """
    Stereo-Kalibrierung mit Schachbrettmuster.
    
    Args:
        image_dir: Ordner mit Bilderpaaren (left_*.jpg, right_*.jpg)
        cols: Anzahl innere Ecken horizontal
        rows: Anzahl innere Ecken vertikal
        square_size_mm: Größe eines Schachbrettfeldes in mm
    """
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    board_size = (cols, rows)

    # 3D Objektpunkte vorbereiten
    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_size_mm

    obj_points = []
    img_points_left = []
    img_points_right = []

    left_images  = sorted(glob.glob(f"{image_dir}/left_*.jpg"))
    right_images = sorted(glob.glob(f"{image_dir}/right_*.jpg"))

    if len(left_images) == 0:
        print("❌ Keine Bilder gefunden! Bitte left_*.jpg und right_*.jpg in den Ordner legen.")
        return None

    print(f"📸 {len(left_images)} Bildpaare gefunden...")
    img_size = None

    for left_path, right_path in zip(left_images, right_images):
        img_l = cv2.imread(left_path, cv2.IMREAD_GRAYSCALE)
        img_r = cv2.imread(right_path, cv2.IMREAD_GRAYSCALE)

        if img_size is None:
            img_size = (img_l.shape[1], img_l.shape[0])

        ret_l, corners_l = cv2.findChessboardCorners(img_l, board_size, None)
        ret_r, corners_r = cv2.findChessboardCorners(img_r, board_size, None)

        if ret_l and ret_r:
            corners_l = cv2.cornerSubPix(img_l, corners_l, (11, 11), (-1, -1), criteria)
            corners_r = cv2.cornerSubPix(img_r, corners_r, (11, 11), (-1, -1), criteria)
            obj_points.append(objp)
            img_points_left.append(corners_l)
            img_points_right.append(corners_r)
            print(f"  ✅ {Path(left_path).name}")
        else:
            print(f"  ⚠️  Schachbrett nicht gefunden: {Path(left_path).name}")

    print(f"\n🔧 Kalibriere mit {len(obj_points)} gültigen Paaren...")

    # Einzelne Kamera-Kalibrierungen
    _, K_l, D_l, _, _ = cv2.calibrateCamera(obj_points, img_points_left, img_size, None, None)
    _, K_r, D_r, _, _ = cv2.calibrateCamera(obj_points, img_points_right, img_size, None, None)

    # Stereo-Kalibrierung
    flags = cv2.CALIB_FIX_INTRINSIC
    _, K_l, D_l, K_r, D_r, R, T, E, F = cv2.stereoCalibrate(
        obj_points, img_points_left, img_points_right,
        K_l, D_l, K_r, D_r, img_size,
        flags=flags,
        criteria=criteria
    )

    # Rektifikation
    R_l, R_r, P_l, P_r, Q, roi_l, roi_r = cv2.stereoRectify(
        K_l, D_l, K_r, D_r, img_size, R, T,
        alpha=0  # 0 = kein schwarzer Rand, 1 = ganzes Bild
    )

    # Rektifikations-Maps vorberechnen
    map_lx, map_ly = cv2.initUndistortRectifyMap(K_l, D_l, R_l, P_l, img_size, cv2.CV_32FC1)
    map_rx, map_ry = cv2.initUndistortRectifyMap(K_r, D_r, R_r, P_r, img_size, cv2.CV_32FC1)

    # Kalibrierungsdaten speichern
    calib_data = {
        "image_size": list(img_size),
        "K_left":  K_l.tolist(),
        "D_left":  D_l.tolist(),
        "K_right": K_r.tolist(),
        "D_right": D_r.tolist(),
        "R": R.tolist(),
        "T": T.tolist(),
        "Q": Q.tolist(),  # Disparität → 3D Tiefe
        "baseline_mm": float(abs(T[0])),
    }

    out_path = Path(image_dir) / "stereo_calibration.json"
    with open(out_path, "w") as f:
        json.dump(calib_data, f, indent=2)

    np.save(Path(image_dir) / "map_lx.npy", map_lx)
    np.save(Path(image_dir) / "map_ly.npy", map_ly)
    np.save(Path(image_dir) / "map_rx.npy", map_rx)
    np.save(Path(image_dir) / "map_ry.npy", map_ry)

    baseline = abs(T[0][0])
    print(f"\n✅ Kalibrierung gespeichert: {out_path}")
    print(f"   📏 Baseline: {baseline:.1f} mm")
    print(f"   🔍 Brennweite (links): fx={K_l[0,0]:.1f}px, fy={K_l[1,1]:.1f}px")

    return calib_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stereo-Kamerakalibrierung")
    parser.add_argument("--images", default="./calib_images", help="Ordner mit Kalibrierungsbildern")
    parser.add_argument("--cols",   type=int,   default=9,    help="Innere Ecken horizontal")
    parser.add_argument("--rows",   type=int,   default=6,    help="Innere Ecken vertikal")
    parser.add_argument("--square", type=float, default=25.0, help="Feldgröße in mm")
    args = parser.parse_args()

    calibrate_stereo(args.images, args.cols, args.rows, args.square)
