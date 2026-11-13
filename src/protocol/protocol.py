from abc import ABC, abstractmethod
from src.data_classes import Classification, ProtocolCtx
import numpy as np


class Protocol(ABC):
    def __init__(self, protocol_params: ProtocolCtx):
        self.protocol_params = protocol_params

    @property
    @abstractmethod
    def localizer(self):
        pass

    @property
    @abstractmethod
    def classifier(self):
        pass

    @abstractmethod
    def predict(self, image: np.ndarray) -> dict[str, Classification]:
        """A abstract class holding the mini-pipeline that takes an objects in an image from 
           localization to classification"""
        pass
