"""

"""
import numpy as np
import cv2 as cv

# Constants for filtering out yellow lines
from matplotlib import pyplot as plt

H_LOWER_Y = 10
S_LOWER_Y = 80
V_LOWER_Y = 150  # 100

H_UPPER_Y = 30
S_UPPER_Y = 255
V_UPPER_Y = 255

# Constants for filtering out white lines
# Thresholding
THRESH_LOWER = 190  # -> 180 optimal
THRESH_UPPER = 255

# Point operation for improving contrast
alpha = 2.5
beta = -300
# Lookup table for improving contrasts
c = np.arange(0, 256)
lookup_table_contrast = np.clip(alpha * c + beta, 0, 255).astype(np.uint8)


# plotting the gradation curve
# x = np.linspace(0, 255, 256)
# y = alpha * x + beta
# y = np.clip(y, 0, 255)
#
# plt.figure()
# plt.plot(x, y)
# plt.title('Gradationskurve')
# plt.show()


class Pipeline:

    @staticmethod
    def extract_yellow_lane(frame):
        """

        """
        new_frame = frame.copy()

        # convert image to HSV
        frame_hsv = cv.cvtColor(new_frame, cv.COLOR_BGR2HSV)

        # filter out the yellow parts of the image
        frame_yellow = cv.inRange(frame_hsv, (H_LOWER_Y, S_LOWER_Y, V_LOWER_Y), (H_UPPER_Y, S_UPPER_Y, V_UPPER_Y))
        new_frame[frame_yellow == 0] = 0
        new_frame[frame_yellow != 0] = 255

        return new_frame

    @staticmethod
    def extract_white_lane2(frame):
        """

        """
        # convert image to grayscale
        frame = cv.cvtColor(frame.copy(), cv.COLOR_RGB2GRAY)

        # improve the contrast of the frame by applying a linear point operation using a lookup table
        improved_contrast = cv.LUT(frame, lookup_table_contrast)

        cv.imshow("Test", improved_contrast)

        # select the white lines/s using thresholding
        ret, new_frame = cv.threshold(improved_contrast, thresh=THRESH_LOWER, maxval=THRESH_UPPER,
                                      type=cv.THRESH_BINARY)
        frame[new_frame == 0] = 0
        frame[new_frame != 0] = 255

        return frame

    @staticmethod
    def extract_white_lane(frame):
        """

        """
        new_frame = np.zeros_like(frame)

        # convert image to grayscale
        frame_hls = cv.cvtColor(frame, cv.COLOR_RGB2HLS)

        # filter out the yellow parts of the image
        frame_white = cv.inRange(frame_hls, (0, 200, 0), (255, 255, 255))
        new_frame[frame_white == 0] = 0
        new_frame[frame_white != 0] = 255

        return new_frame

    @staticmethod
    def canny_edge_detection(frame):
        """

        """
        improvement = False

        if improvement:
            v = np.median(frame)
            sigma = 0.33
            minVal = int(max(0, (1.0 - sigma) * v))
            maxVal = int(min(255, (1.0 + sigma) * v))
        else:
            minVal = 100
            maxVal = 200

        return cv.Canny(frame, minVal, maxVal, edges=True)

    @staticmethod
    def gaussian_blur(frame):
        """

        """

        return cv.GaussianBlur(frame, (5, 5), sigmaX=0, sigmaY=0)

    @staticmethod
    def split_left_right(frame):
        half_width = frame.shape[1] // 2
        width = frame.shape[1]

        frame_left = frame.copy()
        frame_right = frame.copy()

        frame_left[:, half_width:width] = 0
        frame_right[:, 0:half_width] = 0

        return frame_right, frame_left
