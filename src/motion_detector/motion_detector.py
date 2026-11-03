from abc import ABC, abstractmethod

class MotionDetector(ABC):

    @abstractmethod
    def detect_motion(self, frame):
        pass