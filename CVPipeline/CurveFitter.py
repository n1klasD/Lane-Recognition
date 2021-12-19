"""

"""
import numpy as np
import cv2 as cv

from CVPipeline import Pipeline

import config


class CurveFitter:
    def __init__(self):
        self.left_lane_buffer = []
        self.right_lane_buffer = []
        self.left_lifetime = 0
        self.right_lifetime = 0

    def calculateOverlay(self, frame, perspective_transform, final_frame):
        """
        Calculate overlay for frame. Also adaptively check the b values of the curve fit for a threshold
        """

        right, left = Pipeline.split_left_right(frame)
        if right.any() and left.any():
            y1, x1, a1, b1, c1 = CurveFitter.fit_curve_polyfit(right)
            y2, x2, a2, b2, c2 = CurveFitter.fit_curve_polyfit(left)

            self.left_lifetime += 1
            self.right_lifetime += 1

            if self.left_lane_buffer:
                left_newest_curve = self.left_lane_buffer[-1]
                _, _, old_a, old_b, old_c = left_newest_curve
                if (old_b - config.TOLERANCE < b1 < old_b + config.TOLERANCE) or (self.left_lifetime > config.FRAME_LIFETIME):
                    self.left_lifetime = 0
                    self.left_lane_buffer.append((y1, x1, a1, b1, c1))
            else:
                self.left_lane_buffer.append((y1, x1, a1, b1, c1))

            if self.right_lane_buffer:
                right_newest_curve = self.right_lane_buffer[-1]
                _, _, old_a, old_b, old_c = right_newest_curve
                if old_b - config.TOLERANCE < b2 < old_b + config.TOLERANCE or (self.right_lifetime > config.FRAME_LIFETIME):
                    self.right_lifetime = 0
                    self.right_lane_buffer.append((y2, x2, a2, b2, c2))
            else:
                self.right_lane_buffer.append((y2, x2, a2, b2, c2))

            y1, x1, _, _, _ = self.left_lane_buffer[-1]
            y2, x2, _, _, _ = self.right_lane_buffer[-1]

            # retransform the points onto the original image
            points = CurveFitter.stack_points(y1, x1, y2, x2)
            inv_points = perspective_transform.inverse_transform(points)
            return CurveFitter.draw_area(final_frame, inv_points)
        else:
            return final_frame


    @staticmethod
    def fit_curve_polyfit(frame):
        """
        Fit the points of one lane image with a second order polynomial
        """
        w = np.where(frame != 0)
        x = w[1]
        y = w[0]

        yn = np.arange(frame.shape[0])  # x-Koordinaten
        a, b, c = np.polyfit(y, x, 2)
        # print(a, b, c)
        ar = np.poly1d((a, b, c))
        x = np.int32(ar(yn))
        # only return plausible points for the image
        mask = (x >= 0) & (x < frame.shape[1])

        return yn[mask], x[mask], a, b, c

    @staticmethod
    def stack_points(x1, y1, x2, y2):
        A = np.flipud(np.stack((y1, x1), axis=-1))
        B = np.stack((y2, x2), axis=-1)
        C = np.concatenate((A, B))
        return C

    @staticmethod
    def draw_area(frame, points):
        """
        Draw the area on screen, that the points surround
        """
        empty_img = np.zeros_like(frame)
        cv.fillPoly(empty_img, [points.astype(np.int32)], (0, 255, 0))

        return cv.addWeighted(frame, 1, empty_img, 0.3, 0.0)
