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
    source_points = [(470, 0), (220, 400), (790, 400), (590, 0)]
    dest_points = [(290, 0), (290, 400), (630, 400), (630, 0)]
    eye = BirdEye(source_points, dest_points, matrix, dist_coef)
    return eye, cur, capture


def pipeline(img, birdEye, curves, offset):
    ground_image = birdEye.undistort(img)
    binary = preprocess_image(ground_image, np.average(ground_image[150:, 100:-100])+offset)
    wb = np.logical_and(birdEye.sky_view(binary), roi(binary)).astype(np.uint8)
    cv2.imshow("wb", np.copy(wb)*255)
    interest = region_of_interest(wb)
    result = curves.fit(interest)
    ground_img_with_projection = birdEye.project(ground_image, binary,
                                                result['pixel_left_best_fit_curve'],
                                                result['pixel_right_best_fit_curve'])
    position = result['vehicle_position_words']
    cv2.putText(ground_img_with_projection, str(position), (15, 35), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
    return ground_img_with_projection, position


if __name__ == "__main__":
    birdEye, curves, cap = init_all()
    i = 0
    s = 0
    while cap.isOpened():
        try:
            i += 1
            start = time.time()
            ret, frame = cap.read()
            resized = frame[620:-60, 400:-600]
            if np.average(frame) > 80:
                offset = 50
            else:
                offset = 30
            lines_image, position = pipeline(resized, birdEye, curves, offset)
            cv2.imshow("frame", frame)
            cv2.imshow("lines_image", lines_image)
            if cv2.waitKey(1) and keyboard.is_pressed("q"):
                break
            end = time.time()
            if i > 10:
                s += end-start
            print(round(s / (i - 10), 4))
        except:
            print("pass")
    cap.release()
