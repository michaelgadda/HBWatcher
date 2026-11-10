import numpy as np
from datetime import datetime

class Frame:

    def __init__(self, frame_number: int, image: np.ndarray, timestamp: datetime | None = None):
        self.frame_number: int = frame_number
        self.image: np.ndarray = image
        self.timestamp: datetime = timestamp if timestamp is not None else datetime.now()
