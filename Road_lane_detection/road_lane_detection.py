import keyboard
import pyautogui
import numpy as np
import cv2
import pickle
from BirdEye import BirdEye
from lanefilter import LaneFilter
from curves import Curves
import time
from helpers import roi, draw_points, preprocess_image, region_of_interest


def init_all():
    path2video = "video.mov"
    capture = cv2.VideoCapture(path2video)
    cur = Curves(number_of_windows=30, margin=30, minimum_pixels=50,
                 ym_per_pix=30 / 720, xm_per_pix=3.7 / 700)
    calibration_data = pickle.load(open("calibration_data.p", "rb"))
    matrix = calibration_data['camera_matrix']
    dist_coef = calibration_data['distortion_coefficient']
    source_points = [(350 , 0), (120, 400), (700, 400), (500, 0)]
    dest_points = [(260, 0), (260, 400), (580, 400), (580, 0)]
    eye = BirdEye(source_points, dest_points, matrix, dist_coef)
    return eye, capture, cur


def pipeline(img, birdEye, curves):
    ground_image = birdEye.undistort(img)
    binary = preprocess_image(ground_image)
    wb = np.logical_and(birdEye.sky_view(binary), roi(binary)).astype(np.uint8)
    interest = region_of_interest(wb)
    cv2.imshow("sky_view", birdEye.sky_view(binary))
    cv2.imshow("interest", np.copy(interest)*255)
    result = curves.fit(interest)
    ground_img_with_projection = birdEye.project(ground_image, binary,
                                                 result['pixel_left_best_fit_curve'],
                                                 result['pixel_right_best_fit_curve'])
    return ground_img_with_projection


if __name__ == "__main__":
    birdEye, cap, curves = init_all()
    while cap.isOpened():
        try:
            start = time.time()
            ret, frame = cap.read()
            resized = frame[620:-60, 500:-600]
            cv2.imshow("points", draw_points(resized, birdEye.src_points))
            lines_image = pipeline(resized, birdEye, curves)
            cv2.imshow("frame", frame)
            cv2.imshow("lines_image", lines_image)
            if cv2.waitKey(1) and keyboard.is_pressed("q"):
                break
            end = time.time()
            print(end - start)
        except:
            pass
    cap.release()
