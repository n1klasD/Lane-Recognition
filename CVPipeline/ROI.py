import numpy as np
import cv2 as cv


class ROI:
    def __init__(self, width, height, distance_to_middle, bottom_dist_side, top_dist_side):
        self.width = int(width)
        self.height = int(height)

        top_y = height // 2 + distance_to_middle

        self.points = [[top_dist_side, top_y], [self.width - top_dist_side, top_y],
                       [self.width - bottom_dist_side, self.height], [bottom_dist_side, self.height]]

        self.points_np = np.array(self.points, dtype=np.int32)

        self.mask_color = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.mask_color = cv.fillConvexPoly(self.mask_color, self.points_np, (255, 255, 255))
        self.mask_gray = np.zeros((self.height, self.width), dtype=np.uint8)
        self.mask_gray = cv.fillConvexPoly(self.mask_gray, self.points_np, 255)

    def apply_roi(self, img):
        if img.shape == (720, 1280):
            return cv.bitwise_and(self.mask_gray, img)
        else:
            return cv.bitwise_and(self.mask_color, img)


    def draw_roi(self, img):
        img = img.copy()
        return cv.polylines(img, pts=[self.points_np], color=(255, 255, 0), isClosed=True, thickness=3)

# roi = ROI(1280, 720)
# cv.imshow("ROI", roi.mask)
# cv.waitKey(50000)
