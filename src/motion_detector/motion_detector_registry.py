from enum import StrEnum
from src.motion_detector.mog2_motion_detector import Mog2MotionDetector

class MotionDetectorEnum(StrEnum):
    MOG2 = "mog2"

MOTION_DETECTOR_REGISTRY = {MotionDetectorEnum.MOG2: Mog2MotionDetector}

