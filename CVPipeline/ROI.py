import numpy as np
import cv2 as cv


class ROI:
    def __init__(self, width, height, distance_to_middle, bottom_dist_side, top_dist_side):
        self.width = int(width)
        self.height = int(height)

        top_y = height // 2 + distance_to_middle

        self.points = [[top_dist_side, top_y], [self.width - top_dist_side, top_y],
                       [self.width - bottom_dist_side, self.height], [bottom_dist_side, self.height]]

        self.mask = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.mask = cv.fillConvexPoly(self.mask, np.array(self.points, dtype=np.int32), (255, 255, 255))

    def apply_roi(self, img):
        return cv.bitwise_and(self.mask, img)

# roi = ROI(1280, 720)
# cv.imshow("ROI", roi.mask)
# cv.waitKey(50000)
