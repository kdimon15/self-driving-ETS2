import keyboard
import pyautogui
import numpy as np
import cv2
import pickle
from functions import init_birdeye, preprocess_image, detect_lines, draw_lines
import time


def find_lines(image, birdeye):
    normal_img_shape = (image.shape[1], image.shape[0])
    preprocessed_image = preprocess_image(image, birdeye)
    lines = detect_lines(preprocessed_image)
    lines_image = draw_lines(preprocessed_image, lines)
    final_image = birdeye.just_back(normal_img_shape, lines_image)
    return final_image


if __name__ == "__main__":
    path2video = "video.mov"
    cap = cv2.VideoCapture(path2video)
    birdEye = init_birdeye()

    while cap.isOpened():
        start = time.time()
        ret, frame = cap.read()
        resized = frame[590:-60, 500:-600]
        lines_image = find_lines(resized, birdEye)

        cv2.imshow("frame", frame)
        cv2.imshow("lines", lines_image)
        if cv2.waitKey(1) and keyboard.is_pressed("q"):
            break
        end = time.time()
        print(end - start)
        print()
    cap.release()
