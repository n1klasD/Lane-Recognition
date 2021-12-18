"""

"""
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

class CurveFitter:

    @staticmethod
    def fit_curve_polyfit(img):
        new_frame = img.copy()
        w = np.where(new_frame!=0)
        x=w[0]
        y= w[1]

        xn = np.arange(1270 -1)      # x-Koordinaten

        ar = np.poly1d(np.polyfit(y, x, 2))
        y = np.int32(ar(xn))
        mask = (y>=0) & (y<720)
        # [( int(y[i]),x) for i,x in enumerate(xn) if y[i] >=0 and y[i] <= 720]
        return y[mask],xn[mask]