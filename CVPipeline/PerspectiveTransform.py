"""

"""

import cv2 as cv
import numpy as np

import config


class PerspectiveTransform:
    def __init__(self):
        self.source_points = None
        self.dest_points = np.float32(config.destination_points)
        self.transformation_matrix = None
        self.inverse_matrix = None

    def transform(self, frame) -> np.ndarray:

        return cv.warpPerspective(frame, self.transformation_matrix, (frame.shape[1], frame.shape[0]))

    def set_source_points(self, points):
        self.source_points = np.float32(points)
        self.transformation_matrix = cv.getPerspectiveTransform(self.source_points, self.dest_points)
        self.inverse_matrix = cv.getPerspectiveTransform(self.dest_points, self.source_points)

    def inverse_transform(self, points) -> np.ndarray:
        return cv.perspectiveTransform(points, self.inverse_matrix)

    def transform_points(self, points) -> np.ndarray:
        points = np.array([points], dtype=np.float32)
        return cv.perspectiveTransform(points, self.transformation_matrix)


