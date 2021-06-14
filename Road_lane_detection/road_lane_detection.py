import keyboard
import pyautogui
import numpy as np
import cv2
import pickle
from BirdEye import init_birdeye

import time


def pipeline(img, birdEye):
    ground_img = birdEye.undistort(img)
    return img

if __name__ == "__main__":
    path2video = "video.mov"
    cap = cv2.VideoCapture(path2video)
    birdEye = init_birdeye()

    while cap.isOpened():
        start = time.time()
        ret, frame = cap.read()
        resized = frame[590:-60, 500:-600]
        cv2.imshow("resized", resized)
        lines_image = pipeline(resized, birdEye)

        cv2.imshow("frame", frame)
        cv2.imshow("lines", lines_image)
        if cv2.waitKey(1) and keyboard.is_pressed("q"):
            break
        end = time.time()
        print(end - start)
        print()
    cap.release()
