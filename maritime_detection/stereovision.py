import time
import threading
from picamera2 import Picamera2, Preview
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from picamera.cameras import get_frame, camera_thread

frames = {"cam_0": None, "lores_0": None, "cam_1": None, "lores_1": None}
frames_lock = threading.Lock()
running = True
show_window = False

from picamera.cameras import get_frame, stop_cameras


def draw_depth(img0,img1):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''

    img0 = cv.cvtColor(img0, cv.COLOR_BGR2GRAY)
    img1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)


    stereo = cv.StereoSGBM.create(numDisparities=16, blockSize=7)
    #stereo.setTextureThreshold(10)
    stereo.setSpeckleRange(32)
    stereo.setUniquenessRatio(15)
    disparity = stereo.compute(img0,img1)
    return disparity


cv.waitKey(10000)

if __name__ == "__main__":
    t0 = threading.Thread(target=camera_thread, args=(0, "cam_0"), daemon=True)
    t1 = threading.Thread(target=camera_thread, args=(1, "cam_1"), daemon=True)
    t0.start()
    t1.start()
    time.sleep(2.0) # Give cameras a brief window to initialize and spin up
    while True:
        frame_l = get_frame("cam_1")
        frame_r = get_frame("cam_0")

       # ── 1. Objekterkennung zuerst (günstig, ~10ms) ───────────────────────
        lores_l = get_frame("lores_1")
        disparity = draw_depth(frame_r,frame_l)
    
        cv.imshow("camera right", frame_r)
        cv.imshow("camera left", frame_l)
        cv.moveWindow("camera right", 0, 0)
        cv.moveWindow("camera left", 1280, 0)
        gray_disp = cv.merge([disparity, disparity, disparity]).astype(np.uint8)   
        cv.imshow("disparity", gray_disp)
        cv.moveWindow("disparity", 1280, 800)
        #plt.imshow(disparity, cmap='gray')
        #plt.show()

        if cv.waitKey(1) & 0xFF == ord("q"):
            stop_cameras()
            t0.join()
            t1.join()
            break

    cv.destroyAllWindows()
    print("[INFO] Program terminated gracefully.")




