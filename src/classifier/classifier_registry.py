from enum import StrEnum
from src.classifier.yolo26s_classifier import Yolo26sClassifier
from src.classifier.hb_fine_tuned_yolo26s_classifier import HbFtYolo26sClassifier


class ClassifierEnum(StrEnum):
    YOLO26S = "yolo26s"
    HBFTYOLO26S = "hbftyolo26s"



CLSFR_REGISTRY = {
    ClassifierEnum.YOLO26S: Yolo26sClassifier, 
    ClassifierEnum.HBFTYOLO26S: HbFtYolo26sClassifier
}