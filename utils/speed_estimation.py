import math

class SpeedEstimator:
    def __init__(self):
        self.prev_center = None
        self.prev_frame = None

    def update(self, center, frame_index):
        speed = 0.0

        if self.prev_center is not None and self.prev_frame is not None:
            dx = center[0] - self.prev_center[0]
            dy = center[1] - self.prev_center[1]
            distance = math.sqrt(dx*dx + dy*dy)

            frame_diff = frame_index - self.prev_frame
            if frame_diff > 0:
                speed = distance / frame_diff  # pixels per frame

        self.prev_center = center
        self.prev_frame = frame_index

        return speed
