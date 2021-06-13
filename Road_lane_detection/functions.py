import keyboard
import pyautogui
import numpy as np
import cv2
from math import sqrt
import os


def get_screenshot():
    screen = pyautogui.screenshot()
    return screen


def distance(x1, y1, x2, y2):
    return sqrt((x2-x1) ** 2 + (y2-y1)**2)


def remove_thresh(image, thresh=100):
    color_select = np.copy(image)
    red_threshold = thresh
    green_threshold = thresh
    blue_threshold = thresh
    rgb_threshold = [red_threshold, green_threshold, blue_threshold]
    thresholds = (image[:, :, 0] < rgb_threshold[0]) \
                 | (image[:, :, 1] < rgb_threshold[1]) \
                 | (image[:, :, 2] < rgb_threshold[2])
    color_select[thresholds] = [0, 0, 0]
    return color_select


def get_canny(image):  # Поиск границ
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 50, 150)
    return canny


def region_of_interest(image):  # Fill Poly Остается все, что есть в полигоне
    height = image.shape[0]
    width = image.shape[1]
    polygons = np.array([
        [(0, height), (width, height), (500, 0), (250, 0)]
    ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, (255, 255, 255))
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image


def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 7)
    return line_image


def create_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1 * (1 / 5))
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])


def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    l = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)

        # It will fit the polynomial and the intercept and slope
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        if -6 < slope < -0.3:
            left_fit.append((slope, intercept))
        elif 6 > slope > 0.3:
            right_fit.append((slope, intercept))
    if len(left_fit) > 0:
        left_fit_average = np.average(left_fit, axis=0)
        left_line = create_coordinates(image, left_fit_average)
        l.append(left_line)
    if len(right_fit) > 0:
        right_fit_average = np.average(right_fit, axis=0)
        right_line = create_coordinates(image, right_fit_average)
        l.append(right_line)
    return np.array(l)


def make_video():
    out = cv2.VideoWriter("new_output.mov", -1, 15.0, (1920, 1080))
    while True:
        frame = np.array(get_screenshot())
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        out.write(frame)
        if cv2.waitKey(1) and keyboard.is_pressed("q"):
            break
    out.release()


def find_lines(image):
    image = remove_thresh(image)
    cropped_image = region_of_interest(image)
    lane_image = np.copy(cropped_image)
    canny = get_canny(cropped_image)
    cropped_canny_image = region_of_interest(canny)
    lines = cv2.HoughLinesP(cropped_canny_image, 2, np.pi / 180, 100, np.array([]), minLineLength=10, maxLineGap=80)
    all = display_lines(lane_image, lines)
    test_find_lines(canny, np.copy(cropped_image))
    if lines is not None:
        average_lines = average_slope_intercept(image, lines)
        image_square = preprocess_lines(image, average_lines)
    else:
        average_lines = None
    # line_image = display_lines(lane_image, average_lines)
    # combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)
    return cropped_image


def preprocess_lines(image, lines):
    image = image.copy()
    height, width = image.shape[:2]
    if len(lines) == 2:
        x1, y1, x2, y2 = lines[0].reshape(4)
        cv2.line(image, (100, height), (x2, y2), (255, 0, 0), 7)

        x1, y1, x2, y2 = lines[1].reshape(4)
        cv2.line(image, (width-100, height), (x2, y2), (255, 0, 0), 7)
    return image


def test_find_lines(canny_image, image):
    contours = cv2.findContours(canny_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
    # Loop through my contours to find rectangles and put them in a list, so i can view them individually later.
    cntrRect = []
    for i in contours:
        epsilon = 0.05 * cv2.arcLength(i, True)
        approx = cv2.approxPolyDP(i, epsilon, True)
        if True: # len(approx) == 4:
            cv2.drawContours(image, cntrRect, -1, (0, 255, 0), 2)
            cntrRect.append(approx)


if __name__ == "__main__":
    path2video = "output.mov"
    cap = cv2.VideoCapture(path2video)

    while cap.isOpened():
        ret, frame = cap.read()
        resized = frame[550:-60, 600:1320]
        # cv2.imshow("resized", resized)
        lines_image = find_lines(resized)

        cv2.imshow("frame", frame[300:, :-150])
        cv2.imshow("lines", lines_image)
        if cv2.waitKey(1) and keyboard.is_pressed("q"):
            break
    cap.release()