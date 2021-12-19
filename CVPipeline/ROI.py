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

    def apply_roi(self, img, do_crop=True):
        masked = None

        if len(img.shape) == 2:
            masked = cv.bitwise_and(self.mask_gray, img)
        else:
            masked = cv.bitwise_and(self.mask_color, img)

        result = self.crop_to_content(masked) if do_crop else masked
        return self.scale_image(result, scale=0.7)

    def scale_image(self, img, scale=1.0):
        dsize = (int(img.shape[1] * scale), int(img.shape[0] * scale))
        return cv.resize(img, dsize=dsize, interpolation=cv.INTER_CUBIC)

    def crop_to_content(self, img):
        from_x = self.outer_points_np[0][0]
        to_x = self.outer_points_np[1][0]

        from_y = self.outer_points_np[2][1]
        to_y = self.outer_points_np[0][1]

        return img[from_y:to_y, from_x:to_x]

    def reverse(self, img):
        from_x = self.outer_points_np[0][0]
        to_x = self.outer_points_np[1][0]

        from_y = self.outer_points_np[2][1]
        to_y = self.outer_points_np[0][1]

        new_size = (to_x - from_x, to_y - from_y)
        resized = cv.resize(img, dsize=new_size, interpolation=cv.INTER_CUBIC)

        (h, w, *rest) = resized.shape
        new_shape = (self.mask_gray.shape[0], self.mask_gray.shape[1], *rest)
        new_img = np.zeros(new_shape, np.uint8)
        new_img[from_y:to_y, from_x:to_x] = resized

        return new_img

    def draw_roi(self, img):
        img = img.copy()
        return cv.polylines(img, pts=[self.outer_points_np], color=(255, 255, 0), isClosed=True, thickness=3)

# roi = ROI(1280, 720)
# cv.imshow("ROI", roi.mask)
# cv.waitKey(50000)
