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


def draw_points(image, points, d=5, color=(255, 0, 255), make_copy=True):
    if make_copy:
        image = np.copy(image)
    for x, y in points:
        cv2.circle(image, (int(x), int(y)), d, color, cv2.FILLED)
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


def blur_image(image, kernel_size=15):
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


def convert_to_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def white_and_yellow(image):
    lower = np.uint8([170, 170, 170])
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


def preprocess_image(image):
    selected_image = white_and_yellow(image)
    gray = convert_to_gray(selected_image)
    blurred = blur_image(gray, kernel_size=5)
    return blurred
