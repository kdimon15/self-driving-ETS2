import cv2
import numpy as np

def show_dotted_image(this_image, points, name, thickness=5,
                      lines_color=(255, 0, 255), dot_color=(0, 0, 255), d=10):
    image = np.copy(this_image)
    cv2.line(image, (int(points[0][0]), int(points[0][1])), (int(points[1][0]), int(points[1][1])), lines_color, thickness)
    cv2.line(image, (int(points[2][0]), int(points[2][1])), (int(points[3][0]), int(points[3][1])), lines_color, thickness)
    for (x, y) in points:
        cv2.circle(image, (int(x), int(y)), d, dot_color, cv2.FILLED)
    cv2.imshow(name, image)


def preprocess_image(image, birdeye, make_copy=True):
    if make_copy:
        image = np.copy(image)
    image = birdeye.undistort(image, show_dotted=False)
    image = birdeye.sky_view(image, show_dotted=False)
    return image


def draw_points(image, points, d=5, color=(255, 0, 255), make_copy=True):
    if make_copy:
        image = np.copy(image)
    for point in points:
        cv2.circle(image, point, d, color, cv2.FILLED)
    return image


def draw_lines(image, lines, color=(255, 0, 0), thickness=2, make_copy=True):
    if make_copy:
        image = np.copy(image)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(image, (x1, y1), (x2, y2), color, thickness)
    return image


def scale_abs(x, m=255):
    x = np.absolute(x)
    x = np.uint8(m * x / np.max(x))
    return x


def roi(gray, mn=125, mx=1200):
    m = np.copy(gray) + 1
    m[:, :mn] = 0
    m[:, mx:] = 0
    return m
