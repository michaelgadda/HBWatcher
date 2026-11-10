from enum import StrEnum
from src.classifier.yolo26s_classifier import Yolo26sClassifier


class ClassifierEnum(StrEnum):
    YOLO26S = "yolo26s"



CLSFR_REGISTRY = {
    ClassifierEnum.YOLO26S: Yolo26sClassifier

}