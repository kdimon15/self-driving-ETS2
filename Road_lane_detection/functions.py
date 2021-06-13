import keyboard
import pyautogui
import numpy as np
import cv2
from math import sqrt


def get_screenshot():
    screen = pyautogui.screenshot()
    return screen


def make_video():
    out = cv2.VideoWriter("new_output.mov", -1, 15.0, (1920, 1080))
    while True:
        frame = np.array(get_screenshot())
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        out.write(frame)
        if cv2.waitKey(1) and keyboard.is_pressed("q"):
            break
    out.release()


def white_and_yellow(image):
    lower = np.uint8([100, 100, 100])
    upper = np.uint8([255, 255, 255])
    white_mask = cv2.inRange(image, lower, upper)
    # yellow color mask
    lower = np.uint8([190, 190, 0])
    upper = np.uint8([255, 255, 255])
    yellow_mask = cv2.inRange(image, lower, upper)
    # combine the mask
    mask = cv2.bitwise_or(white_mask, yellow_mask)
    masked = cv2.bitwise_and(image, image, mask=mask)
    return masked


def convert_to_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def region_of_interest(image):
    height = image.shape[0]
    width = image.shape[1]
    polygons = np.array([
        [(0, height), (width, height), (500, 0), (250, 0)]
    ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    cv2.imshow("mask", mask)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image


def blur_image(image, kernel_size=15):
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


def detect_edges(image, low_threshold=50, high_threshold=150):
    return cv2.Canny(image, low_threshold, high_threshold)


def find_lines(image):
    selected_image = white_and_yellow(image)
    cv2.imshow("selected", selected_image)
    gray = convert_to_gray(selected_image)
    cv2.imshow("gray", gray)
    blurred_image = blur_image(gray)
    cv2.imshow("blur", blurred_image)
    edge_image = detect_edges(blurred_image, low_threshold=30)
    cv2.imshow("edge", edge_image)
    interest = region_of_interest(edge_image)
    return interest



if __name__ == "__main__":
    path2video = "output.mov"
    cap = cv2.VideoCapture(path2video)

    while cap.isOpened():
        ret, frame = cap.read()
        resized = frame[570:-60, 600:1320]
        # cv2.imshow("resized", resized)
        lines_image = find_lines(resized)

        cv2.imshow("frame", frame[300:, :-150])
        cv2.imshow("lines", lines_image)
        if cv2.waitKey(1) and keyboard.is_pressed("q"):
            break
    cap.release()
