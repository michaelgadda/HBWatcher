import numpy as np
from src.protocol.protocol import Protocol
from src.data_classes import ClassificationCtx, Classification, ProtocolCtx
from src.classifier.yolo26s_classifier import Yolo26sClassifier


class YoloProtocol(Protocol):
    def __init__(self, protocol_params: ProtocolCtx):
        super().__init__(protocol_params)
        self._localizer = None
        self._classifier = Yolo26sClassifier(protocol_params)

    @property
    def localizer(self):
        return self._localizer

    @property
    def classifier(self):
        return self._classifier

    def predict(self, image: np.ndarray) -> dict[str, Classification]:
        classifications = self._classifier.classify(ClassificationCtx(image))
        return classifications
