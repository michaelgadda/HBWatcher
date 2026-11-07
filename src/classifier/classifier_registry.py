from enum import StrEnum
from src.classifier.yolo26s_classifier import Yolo26sClassifier


class CLSFR(StrEnum):
    YOLO26S = "yolo26s"



CLSFR_REGISTRY = {
    CLSFR.YOLO26S: Yolo26sClassifier

}