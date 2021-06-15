import cv2
import numpy as np
import pickle
from helpers import show_dotted_image, draw_lines, draw_points


def init_birdeye():
    calibration_data = pickle.load(open("calibration_data.p", "rb"))
    matrix = calibration_data['camera_matrix']
    dist_coef = calibration_data['distortion_coefficient']
    source_points = [(390, 0), (100, 430), (680, 430), (500, 0)]
    dest_points = [(250, 0), (250, 800), (550, 800), (550, 0)]
    birdEye = BirdEye(source_points, dest_points, matrix, dist_coef)
    return birdEye


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
        temp_image = self.undistort(ground_image, show_dotted=False)
        shape = (temp_image.shape[1], temp_image.shape[0])
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



























