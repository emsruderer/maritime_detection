#!/usr/bin/env python3
"""Record an MP4 video with Picamera2 at 1080p and 12 Mbit/s.

Usage examples:
  python record_mp4_picamera2.py
  python record_mp4_picamera2.py --output my_video.mp4 --duration 15
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
from datetime import date
from pathlib import Path

from picamera2 import Picamera2, Preview
from picamera2.encoders import H264Encoder
from picamera2.outputs import FfmpegOutput

P = "/media/nanno/C876-CC0B/recordings/"  # <-- Anpassen: Pfad zum Speichern der Videos

def parse_args() -> argparse.Namespace:
    d = date.today().isoformat()

    parser = argparse.ArgumentParser(description="Record MP4 via Picamera2")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(f"{P}recording_1080p_12mbit_{d}"),
        help=f"Output MP4 path (default: recording_1080p_12mbit_{d}.mp4)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=3600.0,
        help="Recording time in seconds (default: 3600s = 1 hour)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="Frames per second (default: 10)",
    )   
    parser.add_argument(
        "--shutter",
        type=int,
        default=1000,
        help="shutter speed (default: 1000)",
    )
    parser.add_argument(
        "--hours",
        type=int,
        default=4,
        help="Recording duration in hours (default: 4)",
    )
    parser.add_argument(
        "--preview",
        choices=("auto", "qtgl", "qt", "drm", "none"),
        default="qt",
        help="Preview backend: auto, qtgl, qt, drm, or none (default: qt)",
    )
    parser.add_argument(
        "--pixel-format",
        choices=("YUV420", "RGB888"),
        default="RGB888",
        help="Main stream pixel format (default: RGB888)",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    hour = 0
    width, height = 1920, 1080  # 1080p
    bitrate_bps = 12_000_000  # 12 Mbit/s

    picam2 = Picamera2()
    video_config = picam2.create_video_configuration(
        main={"size": (width, height), "format": args.pixel_format},
        controls={"FrameDurationLimits": (int(1_000_000 / args.fps), int(1_000_000 / args.fps))},
    )
    picam2.configure(video_config)
    picam2.set_controls({"ExposureTime": args.shutter}) 
    min_exp, max_exp, default_exp = picam2.camera_controls["ExposureTime"]
    print(f"ExposureTime set to: {args.shutter} (range: {min_exp} - {max_exp}, default: {default_exp})")
    preview_started = False
    preview_map = {
        "qtgl": Preview.QTGL,
        "qt": Preview.QT,
        "drm": Preview.DRM,
    }
    if args.preview == "auto":
        for name in ("qtgl", "qt", "drm"):
            try:
                picam2.start_preview(preview_map[name], width=1920, height=1080)
                preview_started = True
                print(f"Preview started with backend: {name}")
                break
            except Exception as exc:
                print(f"Preview backend '{name}' unavailable: {exc}")

        if not preview_started:
            print("No preview backend available; continuing without preview.")
    elif args.preview in preview_map:
        selected_preview = args.preview
        if args.pixel_format == "RGB888" and selected_preview == "qtgl":
            print("RGB888 is not supported by qtgl preview. Falling back to qt preview.")
            selected_preview = "qt"
        try:
            picam2.start_preview(preview_map[selected_preview], width=1920, height=1080)
            preview_started = True
            print(f"Preview started with backend: {selected_preview}")
        except Exception as exc:
            print(f"Could not start preview backend '{selected_preview}': {exc}")
            print("Continuing without preview.")

    try:
        while hour < args.hours:
            encoder = H264Encoder(bitrate=bitrate_bps)

            out_path = Path(str(args.output) + f"_{hour}.mp4")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            output = FfmpegOutput(str(out_path))

            should_stop = False

            def _handle_stop(_signum, _frame):
                nonlocal should_stop
                should_stop = True

            signal.signal(signal.SIGINT, _handle_stop)
            signal.signal(signal.SIGTERM, _handle_stop)

            picam2.start_recording(encoder, output)
            start_t = time.time()

            print(f"Recording started: {out_path}")
            print(f"Resolution: {width}x{height} | Bitrate: {bitrate_bps} bps | FPS: {args.fps}")
            print("Press Ctrl+C to stop early...")

            try:
                while not should_stop and (time.time() - start_t) < args.duration:
                    time.sleep(1)
            finally:
                picam2.stop_recording()

            elapsed = time.time() - start_t
            print(f"Recording finished after {elapsed:.2f}s -> {out_path}")
            hour += 1
    finally:
        if preview_started:
            picam2.stop_preview()
    
    input("Recordings finished. Press Enter to exit...")
    picam2.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
