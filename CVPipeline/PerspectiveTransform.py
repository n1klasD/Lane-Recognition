"""

"""

import cv2 as cv
import numpy as np


class PerspectiveTransform:
    def __init__(self):
        self.source_points = [[279, 688], [1026, 688], [684, 450], [596, 450]]
        self.dest_points = [[300, 720], [980, 720], [980, 0], [300, 0]]

    def transform(self, frame) -> np.ndarray:

        src = np.float32(self.source_points)
        dst = np.float32(self.dest_points)

        transformation_matrix = cv.getPerspectiveTransform(src, dst)
        transformed_frame = cv.warpPerspective(frame, transformation_matrix, (frame.shape[1], frame.shape[0]))

        return transformed_frame
