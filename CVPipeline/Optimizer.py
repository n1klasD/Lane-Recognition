import numpy as np
import cv2 as cv

class Optimizer:

    def __init__(self, update_threshold, max_cached_frames=None):
        self.threshold = update_threshold
        self.last_frame = None
        self.since_last_update = 0
        self.max_cached_frames = max_cached_frames

    def needs_update(self, img):
        if self.last_frame is None:
            self.last_frame = img
            return True, 0

        diff_value = self.get_diff(img, self.last_frame)
        should_update = diff_value > self.threshold

        maxcf = self.max_cached_frames
        if maxcf is not None and self.since_last_update >= maxcf:
            should_update = True

        if should_update:
            self.last_frame = img
            self.since_last_update = 0
        else:
            self.since_last_update += 1

        return should_update, diff_value

    def draw_debug_img(self, img):
        not_same = cv.bitwise_xor(self.last_frame, img)
        red = np.zeros((not_same.shape[0], not_same.shape[1], 3), np.uint8)
        red[:] = (0, 0, 255)

        red = cv.bitwise_and(red, red, mask=not_same)
        red = cv.bitwise_or(red, cv.cvtColor(self.last_frame, cv.COLOR_GRAY2RGB))

        diff = self.get_diff(img, self.last_frame)
        text = 'Diff: {:.3f}'.format(diff)
        font = cv.FONT_HERSHEY_SIMPLEX
        text_size, _ = cv.getTextSize(text, font, 1, 2)
        cv.putText(red, text, (20, 20 + text_size[1]), font, 1, (0, 255, 0), 2)
        return red

    def get_diff(self, a, b):
        pixel_count = a.shape[0] * a.shape[1]
        return np.sum(a - b) / pixel_count
