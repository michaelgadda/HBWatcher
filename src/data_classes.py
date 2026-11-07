from dataclasses import dataclass
from enum import StrEnum
from __future__ import annotations
import logging 
import numpy as np

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
        if self.loc_type == LocType.BoundingBox and not self.bbox:
            raise ValueError("Localization Type was bounding_box, but \'bbox\' parameter was not set")
        if self.loc_type == LocType.Pose and not self.pose:
            raise ValueError("Localization Type was pose_estimatino, but \'pose\' parameter was not set")
        if self.loc_type == LocType.Segmentation and not self.segmentation:
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
