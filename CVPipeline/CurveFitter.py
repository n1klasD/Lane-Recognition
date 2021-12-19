"""

"""
import numpy as np
import cv2 as cv

from CVPipeline import Pipeline


class CurveFitter:
    def __init__(self):
        self.left_lane_buffer = []
        self.right_lane_buffer = []
        self.left_lifetime = 0
        self.right_lifetime = 0

    def calculateOverlay(self, frame, perspective_transform, final_frame):
        """
        calculate overlay for frame
        """

        right, left = Pipeline.split_left_right(frame)
        if right.any() and left.any():
            y1, x1, a1, b1, c1 = CurveFitter.fit_curve_polyfit(right)
            y2, x2, a2, b2, c2 = CurveFitter.fit_curve_polyfit(left)

            print("left", self.left_lifetime)
            print("right", self.right_lifetime)
            print("------")
            l = 20

            self.left_lifetime += 1
            self.right_lifetime += 1

            if self.left_lane_buffer:
                left_newest_curve = self.left_lane_buffer[-1]
                _, _, old_a, old_b, old_c = left_newest_curve
                k = 0.5
                if (old_b - k < b1 < old_b + k) or (self.left_lifetime > l):
                    self.left_lifetime = 0
                    self.left_lane_buffer.append((y1, x1, a1, b1, c1))
            else:
                self.left_lane_buffer.append((y1, x1, a1, b1, c1))

            if self.right_lane_buffer:
                right_newest_curve = self.right_lane_buffer[-1]
                _, _, old_a, old_b, old_c = right_newest_curve
                k = 0.5
                if old_b - k < b2 < old_b + k or (self.right_lifetime > l):
                    self.right_lifetime = 0
                    self.right_lane_buffer.append((y2, x2, a2, b2, c2))
            else:
                self.right_lane_buffer.append((y2, x2, a2, b2, c2))

            y1, x1, _, _, _ = self.left_lane_buffer[-1]
            y2, x2, _, _, _ = self.right_lane_buffer[-1]

            points = CurveFitter.stack_points(y1, x1, y2, x2)
            inv_points = perspective_transform.inverse_transform(points)
            return CurveFitter.draw_area(final_frame, inv_points), x1, y1, (a1, b1, c1), x2, y2, (a2, b2, c2)
        else:
            return final_frame, x1, y1, (a1, b1, c1), x2, y2, (a2, b2, c2)


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

    @staticmethod
    def calculate_curvature(fitx, fity, params):
        dep = fitx
        dep_on = fity

        m_per_px_x = 3.7 / 700
        m_per_px_y = 30 / 720

        w2, w1, w0 = np.polyfit(dep_on * m_per_px_y, dep * m_per_px_x, 2)

        second_derivative = 2 * w2
        first_derivative = lambda x: second_derivative * x + w1

        for_x = np.max(dep_on)
        numerator = (1 + first_derivative(for_x) ** 2) ** 1.5
        denominator = abs(second_derivative)

        return numerator / denominator / 10
