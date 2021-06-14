import cv2
import numpy as np
import pickle


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


def init_birdeye():
    calibration_data = pickle.load(open("calibration_data.p", "rb"))
    matrix = calibration_data['camera_matrix']
    dist_coef = calibration_data['distortion_coefficient']
    source_points = [(390, 0), (100, 430), (680, 430), (500, 0)]
    dest_points = [(250, 0), (250, 800), (550, 800), (550, 0)]
    birdEye = BirdEye(source_points, dest_points, matrix, dist_coef)
    return birdEye


def draw_points(image, points, d=5, color=(0, 0, 255), make_copy=True):
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



class BirdEye:
    def __init__(self, pts1, pts2, cam_matrix, distortion_coef):
        self.pts1 = pts1
        self.pts2 = pts2
        self.src_points = np.array(pts1, np.float32)
        self.dest_points = np.array(pts2, np.float32)
        self.warp_matrix = cv2.getPerspectiveTransform(self.src_points, self.dest_points)
        self.inv_warp_matrix = cv2.getPerspectiveTransform(self.dest_points, self.src_points)
        self.cam_matrix = cam_matrix
        self.dist_coef = distortion_coef

    def undistort(self, raw_image, show_dotted=False):
        image = cv2.undistort(raw_image, self.cam_matrix, self.dist_coef, None, self.cam_matrix)
        if show_dotted:
            show_dotted_image(image, self.pts1, "undistort")
        return image

    def sky_view(self, ground_image, show_dotted=False):
        shape = (800, 800)
        warp_image = cv2.warpPerspective(ground_image, self.warp_matrix, shape, flags=cv2.INTER_LINEAR)
        if show_dotted:
            show_dotted_image(warp_image, self.dest_points, "warp")
        return warp_image

    def back_to_normal(self, normal_image, lines, preproc_image, show_dotted=False):
        shape = (normal_image.shape[1], normal_image.shape[0])
        lines_image = np.zeros_like(preproc_image)
        lines_image = draw_lines(lines_image, lines, thickness=7)
        norm_image = cv2.warpPerspective(lines_image, self.inv_warp_matrix, shape, flags=cv2.INTER_LINEAR)
        if show_dotted:
            show_dotted_image(normal_image, self.src_points, "back_to_normal")
        return norm_image

    def just_back(self, shape, image):
        norm_image = cv2.warpPerspective(image, self.inv_warp_matrix, shape, flags=cv2.INTER_LINEAR)
        return norm_image






























