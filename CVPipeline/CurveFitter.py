"""

"""
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


class CurveFitter:

    def __init__(self):
        pass

    def generate_overlay(self, frame):
        pass

    @staticmethod
    def fit_curve_polyfit(frame):
        """
        Fit the points of one lane image with a second order polynomial
        """
        w = np.where(frame != 0)
        x = w[0]
        y = w[1]

        xn = np.arange(frame.shape[1] - 1)  # x-Koordinaten
        g = np.polyfit(y, x, 2)
        ar = np.poly1d(g)
        y = np.int32(ar(xn))
        # only return plausible points for the image
        # todo: Constants
        mask = (y >= 0) & (y < frame.shape[0]) & (y >= 450) & (y < 674)

        return y[mask], xn[mask]

    @staticmethod
    def stack_points(x1, y1, x2, y2):
        A = np.stack((y1, x1), axis=-1)
        B = np.stack((y2, x2), axis=-1)
        C = np.concatenate((A, B))
        return C

    @staticmethod
    def draw_area(frame, x1, y1, x2, y2):
        """
        Draw the area on screen, that the points (x1,y1), (x2,y2) surround
        """
        new_frame = frame.copy()

        C = CurveFitter.stack_points(x1, y1, x2, y2)

        empty_img = np.zeros_like(frame)
        cv.fillPoly(empty_img, [C.astype(np.int32)], (0, 255, 0))

        return cv.addWeighted(new_frame, 1, empty_img, 0.3, 0.0)
