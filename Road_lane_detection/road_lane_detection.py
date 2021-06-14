import keyboard
import pyautogui
import numpy as np
import cv2
import pickle
from birdseye import BirdEye, draw_lines, draw_points, show_dotted_image, init_birdeye
import time


def find_lines(image, birdeye):
    #tmp_image = birdeye.undistort(image, show_dotted=True)
    return image


if __name__ == "__main__":
    path2video = "night_road.mov"
    cap = cv2.VideoCapture(path2video)
    birdEye = init_birdeye()

    while cap.isOpened():
        start = time.time()
        ret, frame = cap.read()
        lines_image = find_lines(frame, birdEye)

        cv2.imshow("frame", frame)
        cv2.imshow("lines", lines_image)
        if cv2.waitKey(1) and keyboard.is_pressed("q"):
            break
        end = time.time()
        print(end - start)
    cap.release()
