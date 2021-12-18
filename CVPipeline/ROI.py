import numpy as np
import cv2 as cv

import config


class ROI:
    #def __init__(self, width, height, distance_to_middle, bottom_dist_side, top_dist_side):
        #self.width = int(width)
        #self.height = int(height)

        #top_y = height // 2 + distance_to_middle

        #self.points = [[top_dist_side, top_y], [self.width - top_dist_side, top_y],
        #               [self.width - bottom_dist_side, self.height], [bottom_dist_side, self.height]]

    def __init__(self):

        # load points from config file
        self.outer_points_np = np.array(config.outer_roi_points, dtype=np.int32)
        self.inner_points_np = np.array(config.inner_roi_points, dtype=np.int32)

        # create color mask
        outer_zeros_color = np.zeros((config.VIDEO_HEIGHT, config.VIDEO_WIDTH, 3), dtype=np.uint8)
        outer_mask_color = cv.fillConvexPoly(outer_zeros_color, self.outer_points_np, (255, 255, 255))

        # add inner mask to outer_mask_color
        mask_color = cv.fillConvexPoly(outer_mask_color, self.inner_points_np, (0, 0, 0))
        self.mask_color = mask_color

        # create grey mask
        outer_zeros_gray = np.zeros((config.VIDEO_HEIGHT, config.VIDEO_WIDTH), dtype=np.uint8)
        outer_mask_gray = cv.fillConvexPoly(outer_zeros_gray, self.outer_points_np, 255)

        # add inner mask to outer_mask_gray
        mask_gray = cv.fillConvexPoly(outer_mask_gray, self.inner_points_np, 0)
        self.mask_gray = mask_gray

    def apply_roi(self, img):
        if len(img.shape) == 2:
            return cv.bitwise_and(self.mask_gray, img)
        else:
            return cv.bitwise_and(self.mask_color, img)

    def draw_roi(self, img):
        img = img.copy()
        return cv.polylines(img, pts=[self.outer_points_np], color=(255, 255, 0), isClosed=True, thickness=3)

# roi = ROI(1280, 720)
# cv.imshow("ROI", roi.mask)
# cv.waitKey(50000)
