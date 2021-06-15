import keyboard
import pyautogui
import numpy as np
import cv2
import pickle
from BirdEye import init_birdeye
from lanefilter import LaneFilter
from curves import Curves
import time
from helpers import roi


def pipeline(img, birdEye, laneFilter, curves):
    ground_image = birdEye.undistort(img)
    binary = laneFilter.apply(ground_image)
    wb = np.logical_and(birdEye.sky_view(binary), roi(binary)).astype(np.uint8)
    result = curves.fit(wb)
    ground_img_with_projection = birdEye.project(ground_image, binary,
                                                 result['pixel_left_best_fit_curve'],
                                                 result['pixel_right_best_fit_curve'])
    return ground_img_with_projection


if __name__ == "__main__":
    path2video = "video.mov"
    cap = cv2.VideoCapture(path2video)
    birdEye = init_birdeye()
    p = {'sat_thresh': 120, "light_thresh": 40, "light_thresh_agr": 205,
         "grad_thresh": (0.7, 1.4), "mag_thresh": 40, "x_thresh": 20}
    laneFilter = LaneFilter(p)
    curves = Curves(number_of_windows=9, margin=10, minimum_pixels=50,
                    ym_per_pix=30/720, xm_per_pix=3.7 / 700)

    while cap.isOpened():
        start = time.time()
        ret, frame = cap.read()
        resized = frame[590:-60, 500:-600]
        lines_image = pipeline(resized, birdEye, laneFilter, curves)

        cv2.imshow("frame", frame)
        cv2.imshow("lines_image", lines_image)
        if cv2.waitKey(1) and keyboard.is_pressed("q"):
            break
        end = time.time()
        print(end - start)
        print()
    cap.release()
