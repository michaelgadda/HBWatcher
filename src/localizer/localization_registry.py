from enum import StrEnum
from __future__ import annotations

from src.localizer.yolo26s_localizer import Yolo26sLocalizer

class LocalizerEnum(StrEnum):
    YOLO26S = "yolo26s"

LOC_REGISTRY = {
    LocalizerEnum.YOLO26S: Yolo26sLocalizer
}