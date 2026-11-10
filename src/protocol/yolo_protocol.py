import numpy as np
from src.protocol.protocol import Protocol
from src.data_classes import ClassificationCtx
from src.classifier.yolo26s_classifier import Yolo26sClassifier


class YoloProtocol(Protocol):
    def __init__(self):
        self._localizer = None
        self._classifier = Yolo26sClassifier()

    @property
    def localizer(self):
        return self._localizer

    @property
    def classifier(self):
        return self._classifier

    def predict(self, image: np.ndarray):
        classifications = self._classifier.classify(ClassificationCtx(image))
        return classifications
