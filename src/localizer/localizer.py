from abc import ABC, abstractmethod
import numpy as np
from data_classes import Localization

class Localizer(ABC):

    @abstractmethod
    def localize(self, image: np.ndarray) -> Localization:
        """A abstract method to localize objects in an image"""
        pass
