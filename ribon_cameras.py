import threading
import time
import cv2
from picamera2 import Picamera2, controls
import libcamera
import numpy as np
import argparse
import time
from pathlib import Path
import sys
from ultralytics import YOLO

# 1. Initialize YOLO model (Optimized NCNN format)
# Ensure the folder path 'yolov8n_ncnn' matches your exported directory
model = YOLO("yolov8n_ncnn_model")

# Thread-safe global holders for camera frames
frames = {"cam0": None, "cam1": None}
running = True

def camera_thread(camera_id, frame_key):
    """Handles the libcamera feed for an individual ribbon port."""
    global running, frames
    try:
        # Open Picamera2 for a specific camera index (0 or 1)
        picam = Picamera2(camera_num=camera_id)
        
        # Configure configuration for optimal speed vs quality
        # Lower resolution drastically improves YOLO pipeline FPS
            # For cam0, we can use a higher resolution for better detection quality
        config = picam.create_preview_configuration(main={"size": (1280, 720)})
        picam.configure(config)
        picam.start()
        
        print(f"[INFO] Camera {camera_id} started successfully.")
        
        while running:
            # Capture individual array frames from the pipeline
            frame = picam.capture_array()
            # Picamera2 outputs RGB, OpenCV expects BGR for displaying
            frames[frame_key] = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            time.sleep(0.02) # Yield to prevent CPU core pegging
            
    except Exception as e:
        print(f"[ERROR] Camera {camera_id} failed: {e}")
    finally:
        picam.stop()

# 2. Launch concurrent threads for both ribbon connectors
t0 = threading.Thread(target=camera_thread, args=(0, "cam0"), daemon=True)
t1 = threading.Thread(target=camera_thread, args=(1, "cam1"), daemon=True)
t0.start()
t1.start()

# Give cameras a brief window to initialize and spin up
time.sleep(2.0)

print("[INFO] Processing streams... Press 'q' to exit.")

# 3. Main processing and inference loop
while True:
    img0 = frames["cam0"]
    img1 = frames["cam1"]
    
    # Process Camera 0 if frame is ready
    if img0 is not None:
        results0 = model(img0, stream=True, verbose=False)
        for r in results0:
            print(r)
            img0 = r.plot() # Draw YOLO results on the frame
        cv2.imshow("Ribbon Camera 0 - YOLO", img0)
        cv2.moveWindow("Ribbon Camera 0 - YOLO", 500, 100) # Position window for cam0
 
    # Process Camera 1 if frame is ready
    if img1 is not None:
        results1 = model(img1, stream=True, verbose=False)
        for r in results1:
            img1 = r.plot() 
        cv2.imshow("Ribbon Camera 1 - YOLO", img1)
        cv2.moveWindow("Ribbon Camera 1 - YOLO", 500, 700) # Position window for cam1
    # Break loop safely with the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        running = False
        break

cv2.destroyAllWindows()
print("[INFO] Program terminated gracefully.")
