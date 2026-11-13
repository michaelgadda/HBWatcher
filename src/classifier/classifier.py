# An abstract class for handling pure classifiers (Need to figure out how to handle multi-head (localization & classification))
from abc import ABC, abstractmethod
from src.data_classes import Classification, ClassificationCtx, ProtocolCtx

class Classifier(ABC):
    def __init__(self, protocol_params: ProtocolCtx):
        self.protocol_params = protocol_params

    @abstractmethod
    def classify(self, context: ClassificationCtx) -> dict[str, Classification]:
        """A method to classify (an) object(s)
            The output is a dictionary of 'Classification'(s) so we can have O(1)
            look ups when hunting for specific class types (in this case bird, hummingbird, or type of hummingbird)
        """
        pass
