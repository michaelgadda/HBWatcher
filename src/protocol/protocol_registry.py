from __future__ import annotations

from enum import StrEnum
from collections import namedtuple
from src.classifier.yolo26s_classifier import Yolo26sClassifier
from src.classifier.hb_fine_tuned_yolo26s_classifier import HbFtYolo26sClassifier

class ProtocolEnum(StrEnum):
    YOLO_PROTOCOL = "yolo_protocol"
    FT_HB_YOLO_PROTOCOL = "ft_hb_yolo_protocol"

# Lazy dataclass...
InfPipe = namedtuple("InfPipe", ["loc", "clsf"]) 

PROTOCOL_REGISTRY = {
    ProtocolEnum.YOLO_PROTOCOL : InfPipe(None, Yolo26sClassifier),
    ProtocolEnum.FT_HB_YOLO_PROTOCOL : InfPipe(None, HbFtYolo26sClassifier),
}