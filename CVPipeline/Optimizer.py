import numpy as np

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

    def get_diff(self, a, b):
        pixel_count = a.shape[0] * a.shape[1]
        return np.sum(a - b) / pixel_count
