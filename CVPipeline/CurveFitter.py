"""

"""
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

from CVPipeline import Pipeline


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
    def draw_area(frame, x1, y1, x2, y2):
        """
        Draw the area on screen, that the points (x1,y1), (x2,y2) surround
        """
        new_frame = frame.copy()

        C = CurveFitter.stack_points(x1, y1, x2, y2)

        empty_img = np.zeros_like(frame)
        cv.fillPoly(empty_img, [C.astype(np.int32)], (0, 255, 0))

        return cv.addWeighted(new_frame, 1, empty_img, 0.3, 0.0)

    @staticmethod
    def hough_lines(frame, normal_frame):
        # This returns an array of r and theta values

        lines = cv.HoughLines(frame, 1, np.pi / 180, 50)
        frame = cv.cvtColor(frame, cv.COLOR_GRAY2RGB)
        normal_frame = cv.cvtColor(normal_frame, cv.COLOR_GRAY2RGB)

        # The below for loop runs till r and theta values
        # are in the range of the 2d array
        if lines is not None:
            r, theta = lines[0][0]
            # Stores the value of cos(theta) in a
            a = np.cos(theta)

            # Stores the value of sin(theta) in b
            b = np.sin(theta)

            # x0 stores the value rcos(theta)
            x0 = a * r

            # y0 stores the value rsin(theta)
            y0 = b * r

            # x1 stores the rounded off value of (rcos(theta)-1000sin(theta))
            x1 = int(x0 + 1000 * (-b))

            # y1 stores the rounded off value of (rsin(theta)+1000cos(theta))
            y1 = int(y0 + 1000 * (a))

            # x2 stores the rounded off value of (rcos(theta)+1000sin(theta))
            x2 = int(x0 - 1000 * (-b))

            # y2 stores the rounded off value of (rsin(theta)-1000cos(theta))
            y2 = int(y0 - 1000 * (a))
            # cv.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

            # Geradengleichungen

            k = 150

            l1_x1 = x1 - k
            l1_y1 = y1
            l1_x2 = x2 - k
            l1_y2 = y2
            cv.line(normal_frame, (l1_x1, l1_y1), (l1_x2, l1_y2), (0, 255, 0), 2)

            l2_x1 = x1 + k
            l2_y1 = y1
            l2_x2 = x2 + k
            l2_y2 = y2
            cv.line(normal_frame, (l2_x1, l2_y1), (l2_x2, l2_y2), (0, 255, 0), 2)

            m1, c1 = CurveFitter.getLineParameters(l1_x1, l1_y1, l1_x2, l1_y2)
            m2, c2 = CurveFitter.getLineParameters(l2_x1, l2_y1, l2_x2, l2_y2)

            left, right = Pipeline.split_left_right(normal_frame)
            y_poly, x_poly = CurveFitter.fit_curve_polyfit(right)

            # select only the points, that are within the boundarys

            mask1 = (x_poly >= (m1 * y_poly + c1)) & (x_poly <= (m2 * y_poly + c2))
            mask2 = (x_poly <= (m1 * y_poly + c1)) & (x_poly >= (m2 * y_poly + c2))

            mask = mask1 | mask2
            normal_frame[y_poly[mask], x_poly[mask]] = (0, 0, 255)
            # normal_frame[x_poly, y_poly] = (0, 0, 255)

            # adjust frame

        return normal_frame

    @staticmethod
    def getLineParameters(x1, y1, x2, y2):
        m = (y2 - y1) / (x2 - x1)
        c = -m * x1 + y1

        return m, c
