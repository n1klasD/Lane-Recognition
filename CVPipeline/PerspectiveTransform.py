"""

"""

import cv2 as cv
import numpy as np

import config


class PerspectiveTransform:
    def __init__(self):
        self.source_points = None
        self.dest_points = config.destination_points

    def transform(self, frame) -> np.ndarray:

        src = np.float32(self.source_points)
        dst = np.float32(self.dest_points)

        transformation_matrix = cv.getPerspectiveTransform(src, dst)
        transformed_frame = cv.warpPerspective(frame, transformation_matrix, (frame.shape[1], frame.shape[0]))

        return transformed_frame

    def set_source_points(self, points):
        self.source_points = points
