from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import logging
import numpy as np
import cv2
from src.motion_detector.motion_detector_registry import MotionDetectorEnum

LOGGER = logging.getLogger(__name__)

class LocType(StrEnum):
    BoundingBox = "bounding_box"
    Pose = "pose_estimation"
    Segmentation = "segmentation"

@dataclass
class Localization:
    loc_type: LocType
    bbox: BBox | None = None
    pose: list[Point2D | Point3D] | None = None
    segmentation: np.ndarray | None = None

    def __post_init__(self):
        if self.loc_type == LocType.BoundingBox and self.bbox is None:
            raise ValueError("Localization Type was bounding_box, but \'bbox\' parameter was not set")
        if self.loc_type == LocType.Pose and self.pose is None:
            raise ValueError("Localization Type was pose_estimatino, but \'pose\' parameter was not set")
        if self.loc_type == LocType.Segmentation and self.segmentation is None:
            raise ValueError("Localization Type was segmentation, but \'segmentation\' parameter was not set")

@dataclass
class ClassificationCtx:
    image: np.ndarray
    localization: Localization | None = None

    def __post_init__(self):
        if not self.localization:
            LOGGER.warning("Localization has not been done; " \
            "meaning localization needs to be in the classification call.")

@dataclass
class Classification:
    classification: str
    localization: Localization

@dataclass
class BBox:
    xmin: float
    ymin: float
    xmax: float
    ymax: float

@dataclass
class Point2D:
    x: float
    y: float

@dataclass
class Point3D:
    x: float
    y: float
    z: float

@dataclass
class StreamCtx:
    capture_path: str
    inference_persistence: int = 30
    motion_detector: MotionDetectorEnum | None = MotionDetectorEnum.MOG2
    backend: int = cv2.CAP_FFMPEG

@dataclass
class ProtocolCtx:
    iou: float = .2
    conf: float = .8
