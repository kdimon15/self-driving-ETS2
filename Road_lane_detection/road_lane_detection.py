import keyboard
import pyautogui
import numpy as np
import cv2
import pickle
from functions import BirdEye, draw_lines, draw_points, show_dotted_image, init_birdeye
import time


def find_lines(image, birdeye):
    tmp_image = birdeye.undistort(image, show_dotted=False)
    tmp_image = birdeye.sky_view(tmp_image, show_dotted=False)
    return tmp_image


if __name__ == "__main__":
    path2video = "night_road.mov"
    cap = cv2.VideoCapture(path2video)
    birdEye = init_birdeye()

    while cap.isOpened():
        start = time.time()
        ret, frame = cap.read()
        resized = frame[450:-60, 300:-450]
        #cv2.imshow("resized", resized)
        lines_image = find_lines(resized, birdEye)

        # cv2.imshow("frame", frame)
        cv2.imshow("lines", lines_image)
        if cv2.waitKey(1) and keyboard.is_pressed("q"):
            break
        end = time.time()
        print(end - start)
        print()
    cap.release()
