from abc import ABC, abstractmethod
import numpy as np
from data_classes import Localization, ProtocolCtx


class Localizer(ABC):
    def __init__(self, protocol_params: ProtocolCtx):
        self.protocol_params = protocol_params

    @abstractmethod
    def localize(self, image: np.ndarray) -> list[Localization]:
        """A abstract method to localize objects in an image"""
        pass
