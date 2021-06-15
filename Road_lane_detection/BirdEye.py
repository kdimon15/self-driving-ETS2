import cv2
import numpy as np
import pickle
from helpers import show_dotted_image, draw_lines, draw_points


def init_birdeye():
    calibration_data = pickle.load(open("calibration_data.p", "rb"))
    matrix = calibration_data['camera_matrix']
    dist_coef = calibration_data['distortion_coefficient']
    source_points = [(390, 0), (100, 430), (680, 430), (500, 0)]
    dest_points = [(170, 0), (170, 800), (600, 800), (600, 0)]
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

    def project(self, ground_image, sky_lane, left_fit, right_fit, color=(0, 255, 0)):
        z = np.zeros_like(sky_lane)
        sky_lane = np.dstack((z, z, z))

        kl, kr = left_fit, right_fit
        h = sky_lane.shape[0]
        ys = np.linspace(0, h-1, h)
        lxs = kl[0] * (ys ** 2) + kl[1] * ys + kl[2]
        rxs = kr[0] * (ys ** 2) + kr[1] * ys + kr[2]

        pts_left = np.array([np.transpose(np.vstack([lxs, ys]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([rxs, ys])))])
        pts = np.hstack((pts_left, pts_right))

        cv2.fillPoly(sky_lane, np.int_(pts), color)

        shape = (sky_lane.shape[1], sky_lane.shape[0])
        ground_lane = cv2.warpPerspective(sky_lane, self.inv_warp_matrix, shape)

        result = cv2.addWeighted(ground_image, 1, ground_lane, 0.3, 0)
        return result



























