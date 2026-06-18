import threading
import time
import cv2 as cv
from picamera2 import Picamera2, controls
import libcamera
import numpy as np
import argparse
import time
from pathlib import Path
import sys
from ultralytics import YOLO
from picamera.cameras import get_frame, camera_thread, stop_cameras

# 1. Initialize YOLO model (Optimized NCNN format)
# Ensure the folder path 'yolov8n_ncnn' matches your exported directory
model = YOLO("yolov8n_ncnn_model")


if __name__ == "__main__":
    t0 = threading.Thread(target=camera_thread, args=(0, "cam_0"), daemon=True)
    t1 = threading.Thread(target=camera_thread, args=(1, "cam_1"), daemon=True)
    t0.start()
    t1.start()
    time.sleep(2.0) # Give cameras a brief window to initialize and spin upq
    while True:
        frame_l = get_frame("cam_1")
        frame_r = get_frame("cam_0")
        
        # Process Camera 0 if frame is ready
        if frame_r is not None:
            results0 = model(frame_r, stream=True, verbose=False)
            for r in results0:
                print(r)
                img0 = r.plot() # Draw YOLO results on the frame
            cv.imshow("Camera Right", img0)
            cv.moveWindow("Camera Right", 0, 0) # Position window for camera right
     
        # Process Camera 1 if frame is ready
        if frame_l is not None:
            results1 = model(frame_l, stream=True, verbose=False)
            for r in results1:
                img1 = r.plot() 
            cv.imshow("Camera Left", img1)
            cv.moveWindow("Camera Left", 1283, 0) # Position window for camera left
        # Break loop safely with the 'q' key
        if cv.waitKey(1) & 0xFF == ord("q"):
            stop_cameras()
            t0.join()
            t1.join()
            break

    cv.destroyAllWindows()
    print("[INFO] Program terminated gracefully.")
