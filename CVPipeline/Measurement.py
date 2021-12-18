import cv2 as cv
import numpy as np
from time import time_ns

class Measurement:

    def __init__(self, target_time=None):
        self.last_measurement = 0
        self.fps_history = []
        self.target_time = target_time
        self.frame_timing = self.measure('Frame')

    def beginFrame(self):
        self.frame_timing.reset()

    def endFrame(self):
        self.frame_timing.finish()

    def drawFrameTiming(self, img):
        milliseconds = self.frame_timing.toMs()
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
        font = cv.FONT_HERSHEY_SIMPLEX

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

    def drawTiming(self, img, timing, row):
        if not timing.finished:
            raise Error('Attempted to draw active timing')

        font=cv.FONT_HERSHEY_SIMPLEX

        text = timing.getMsString()
        text_size, _ = cv.getTextSize(text, font, 1, 1)
        text_x = img.shape[1] - text_size[0] - 20
        text_y = img.shape[0] - 20 - text_size[1] * (row + 1) - 10 * row
        color = (0, 0, 255)

        pos = (text_x, text_y)
        cv.putText(img, text, pos, font, 1, color, 2)

    def measure(self, name):
        return Timing(name)

class Timing:

    def __init__(self, name):
        self.start = time_ns()
        self.name = name

    def reset(self):
        self.start = time_ns()
        self.finished = False

    def finish(self):
        self.end = time_ns()
        self.finished = True

    def toMs(self):
        return (self.end - self.start) / (10 ** 6)

    def getMsString(self):
        return '{}: {:.3f} ms'.format(self.name, self.toMs())
