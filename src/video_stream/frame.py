import numpy as np
from datetime import datetime, now

class Frame:

    def __init__(self, frame_number: int, image: np.ndarray, timestamp: datetime = now()):
        self.frame_number: int = frame_number
        self.image: np.ndarray = image
        self.timestamp: datetime = timestamp