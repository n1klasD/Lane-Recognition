import cv2 as cv
import numpy as np
from time import time_ns

class Measurement:

    def __init__(self, target_time=None):
        self.start = 0
        self.last_measurement = 0
        self.fps_history = []
        self.target_time = target_time

    def begin(self):
        self.start = time_ns()

    def end(self):
        self.last_measurement = time_ns() - self.start

    def drawToImage(self, img):
        milliseconds = self.last_measurement / (10 ** 6)
        fps = 1000 / milliseconds

        if len(self.fps_history) >= 5:
            self.fps_history = np.roll(self.fps_history, 1)
            self.fps_history[0] = fps
        else:
            self.fps_history.insert(0, fps)

        avg_fps = np.average(self.fps_history)
        time_str = '{:.1f} ms'.format(milliseconds)
        fps_str = '{:.0f} FPS'.format(avg_fps)

        if self.target_time is not None and milliseconds > self.target_time:
            self.drawText(img, time_str, 0, color=(0, 0, 255))
            self.drawCircle(img, 0, (0, 0, 255))
        else:
            self.drawText(img, time_str, 0)

        self.drawText(img, fps_str, 1)

    def drawText(self, img, text, row, color=(0, 255, 200)):
        font=cv.FONT_HERSHEY_SIMPLEX

        text_size, _ = cv.getTextSize(text, font, 1, 1)
        text_x = img.shape[1] - text_size[0] - 20
        text_y = text_size[1] * (row + 1) + 10 * row + 20

        pos = (text_x, text_y)
        cv.putText(img, text, pos, font, 1, color, 2)

    def drawCircle(self, img, row, color):
        radius = 22

        pos = (
            radius + 20,
            radius * (row + 1) + 10 * row + 20
        )

        cv.circle(img, pos, radius, color, -1)
