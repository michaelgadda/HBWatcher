from abc import ABC, abstractmethod
from src.data_classes import Classification
import numpy as np


class Protocol(ABC):

    @property
    @abstractmethod
    def localizer(self):
        pass

    @property
    @abstractmethod
    def classifier(self):
        pass

    @abstractmethod
    def predict(self, image: np.ndarray) -> list[Classification]:
        """A abstract class holding the mini-pipeline that takes an objects in an image from 
           localization to classification"""
        pass
