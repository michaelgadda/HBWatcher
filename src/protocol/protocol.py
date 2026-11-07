from abc import ABC, abstractmethod
from src.data_classes import Classification
import numpy as np


class Protocol(ABC):

    @abstractmethod
    def predict(image: np.ndarray) -> list[Classification]:
        """A abstract class holding the mini-pipeline that takes an objects in an image from 
           localization to classification"""
        pass
