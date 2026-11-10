from __future__ import annotations

from enum import StrEnum
from src.protocol.yolo_protocol import YoloProtocol

class ProtocolEnum(StrEnum):
    YOLO_PROTOCOL = "yolo_protocol"


PROTOCOL_REGISTRY = {
    ProtocolEnum.YOLO_PROTOCOL : YoloProtocol
}