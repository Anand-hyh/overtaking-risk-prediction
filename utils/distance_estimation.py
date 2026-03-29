class DistanceEstimator:
    def __init__(self):
        # Thresholds based on bounding box width (pixels)
        # You can tune these depending on video resolution
        self.very_close_threshold = 250
        self.near_threshold = 150
        self.mid_threshold = 70

    def estimate(self, box_width):
        """
        Estimate relative distance category from bounding box width.
        Larger bounding box -> object is closer.
        """

        if box_width > self.very_close_threshold:
            return "VERY CLOSE"

        elif box_width > self.near_threshold:
            return "NEAR"

        elif box_width > self.mid_threshold:
            return "MID"

        else:
            return "FAR"