import keyboard
import pyautogui
import numpy as np
import cv2
import pickle
from BirdEye import init_birdeye
from lanefilter import LaneFilter
import time
from helpers import roi


def pipeline(img, birdEye, laneFilter):
    undistorted_image = birdEye.undistort(img)
    binary = laneFilter.apply(undistorted_image)
    cv2.imshow("undistorted", undistorted_image)
    cv2.imshow("binary", binary)
    lane_pixels = np.logical_and(birdEye.sky_view(binary), roi(binary))
    return lane_pixels

if __name__ == "__main__":
    path2video = "video.mov"
    cap = cv2.VideoCapture(path2video)
    birdEye = init_birdeye()
    p = {'sat_thresh': 50, "light_thresh": 10, "light_thresh_agr": 100,
         "grad_thresh": (0.7, 1.4), "mag_thresh": 10, "x_thresh": 10}
    laneFilter = LaneFilter(p)

    while cap.isOpened():
        start = time.time()
        ret, frame = cap.read()
        resized = frame[590:-60, 500:-600]
        #cv2.imshow("resized", resized)
        lines_image = pipeline(resized, birdEye, laneFilter)

        #cv2.imshow("frame", frame)
        if cv2.waitKey(1) and keyboard.is_pressed("q"):
            break
        end = time.time()
        print(end - start)
        print()
    cap.release()
