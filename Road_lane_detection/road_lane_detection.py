import keyboard
import pyautogui
import numpy as np
import cv2
import pickle
from BirdEye import BirdEye
from lanefilter import LaneFilter
from curves import Curves
import time
from helpers import roi, draw_points, preprocess_image


def init_all():
    path2video = "video.mov"
    capture = cv2.VideoCapture(path2video)
    p = {'sat_thresh': 120, "light_thresh": 40, "light_thresh_agr": 210,
         "grad_thresh": (0.7, 1.4), "mag_thresh": 40, "x_thresh": 20}
    l_filter = LaneFilter(p)
    cur = Curves(number_of_windows=30, margin=30, minimum_pixels=50,
                 ym_per_pix=30 / 720, xm_per_pix=3.7 / 700)
    calibration_data = pickle.load(open("calibration_data.p", "rb"))
    matrix = calibration_data['camera_matrix']
    dist_coef = calibration_data['distortion_coefficient']
    source_points = [(350, 0), (120, 400), (750, 400), (520, 0)]
    dest_points = [(206, 0), (206, 400), (634, 400), (634, 0)]
    eye = BirdEye(source_points, dest_points, matrix, dist_coef)
    return eye, capture, l_filter, cur


def pipeline(img, birdEye, laneFilter, curves):
    ground_image = birdEye.undistort(img)
    binary = preprocess_image(ground_image)
    wb = np.logical_and(birdEye.sky_view(binary), roi(binary)).astype(np.uint8)
    result = curves.fit(wb)
    ground_img_with_projection = birdEye.project(ground_image, binary,
                                                 result['pixel_left_best_fit_curve'],
                                                 result['pixel_right_best_fit_curve'])
    return ground_img_with_projection


# def pipeline(img, birdEye, laneFilter, curves):
#     ground_image = birdEye.undistort(img)
#     binary = preprocess_image(ground_image)
#     #binary = laneFilter.apply(ground_image)
#     wb = np.logical_and(birdEye.sky_view(binary), roi(binary)).astype(np.uint8)
#     result = curves.fit(wb)
#     ground_img_with_projection = birdEye.project(ground_image, binary,
#                                                  result['pixel_left_best_fit_curve'],
#                                                  result['pixel_right_best_fit_curve'])
#     return ground_img_with_projection


if __name__ == "__main__":
    birdEye, cap, laneFilter, curves = init_all()
    while cap.isOpened():
        try:
            start = time.time()
            ret, frame = cap.read()
            resized = frame[620:-60, 500:-600]
            lines_image = pipeline(resized, birdEye, laneFilter, curves)
            cv2.imshow("frame", frame)
            cv2.imshow("lines_image", lines_image)
            if cv2.waitKey(1) and keyboard.is_pressed("q"):
                break
            end = time.time()
            print(end - start)
        except:
            pass
    cap.release()
