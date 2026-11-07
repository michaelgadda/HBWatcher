# An abstract class for handling pure classifiers (Need to figure out how to handle multi-head (localization & classification))
from abc import ABC, abstractmethod
from src.data_classes import Classification, ClassificationCtx

class Classifier(ABC):

    @abstractmethod
    def classify(self, context: ClassificationCtx) -> list[Classification]:
        """A method to classify (an) object(s)
        (TBD) The first output will be a **list of cls name(s) associatated with
        the second param, which will be the bounding boxes (localization) of each 
        classification. 
        """
        pass
