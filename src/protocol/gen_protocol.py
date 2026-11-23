import numpy as np
from src.protocol.protocol import Protocol
from src.classifier.classifier import Classifier
from src.data_classes import ClassificationCtx, Classification, ProtocolCtx, Localization
from src.localizer.localizer import Localizer
from src.protocol.protocol_registry import PROTOCOL_REGISTRY


class GenericProtocol(Protocol): #TODO: Leaving abstract class for time being, but likely useless.
    def __init__(self, protocol_params: ProtocolCtx):
        super().__init__(protocol_params)
        tps = PROTOCOL_REGISTRY[protocol_params.protocol_name]
        self._localizer: Localizer | None = tps.loc(protocol_params) if tps.loc else None
        self._classifier: Classifier = tps.clsf(protocol_params)

    @property
    def localizer(self):
        return self._localizer

    @property
    def classifier(self):
        return self._classifier

    def predict(self, image: np.ndarray) -> dict[str, list[Classification]]:
        clsfctx = ClassificationCtx(image)
        if self._localizer:
            self._localizer: Localizer
            localizations: list[Localization] = self._localizer.localize()
            clsfctx.localization = localizations
        classifications = self._classifier.classify(clsfctx)
        return classifications
