"""

"""
import numpy as np
import cv2 as cv


class CurveFitter:

    @staticmethod
    def fit_curve_polyfit(frame):
        """
        Fit the points of one lane image with a second order polynomial
        """
        w = np.where(frame != 0)
        x = w[1]
        y = w[0]

        yn = np.arange(frame.shape[0])  # x-Koordinaten
        g = np.polyfit(y, x, 2)
        ar = np.poly1d(g)
        x = np.int32(ar(yn))
        # only return plausible points for the image
        # todo: Constants
        mask = (x >= 0) & (x < frame.shape[1])  # & (y >= 450) & (y < 674)

        return yn[mask], x[mask]

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
