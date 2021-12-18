"""

"""
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


class CurveFitter:

    @staticmethod
    def fit_curve_polyfit(img):
        new_frame = img.copy()
        w = np.where(new_frame != 0)
        x = w[0]
        y = w[1]

        xn = np.arange(1270 - 1)  # x-Koordinaten

        ar = np.poly1d(np.polyfit(y, x, 2))
        y = np.int32(ar(xn))
        mask = (y >= 0) & (y >= 450) & (y < 720)
        # [( int(y[i]),x) for i,x in enumerate(xn) if y[i] >=0 and y[i] <= 720]
        return y[mask], xn[mask]

    @staticmethod
    def draw_area(frame, x1, y1, x2, y2):
        new_frame = frame.copy()

        A = np.stack((y1, x1), axis=-1)
        B = np.stack((y2, x2), axis=-1)
        C = np.concatenate((A, B))

        empty_img = np.zeros_like(frame)
        cv.fillPoly(empty_img, [C.astype(np.int32)], (0, 255, 0))

        return cv.addWeighted(new_frame, 1, empty_img, 0.5, 0.0)
