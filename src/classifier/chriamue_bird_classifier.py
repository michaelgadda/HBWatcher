from src.classifier.classifier import Classifier
from src.data_classes import Classification, ClassificationCtx, \
    ProtocolCtx
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification


class ChriamueBirdClassifier(Classifier):
    def __init__(self, protocol_params: ProtocolCtx):
        super().__init__(protocol_params)
        self._extractor = AutoImageProcessor.from_pretrained("chriamue/bird-species-classifier")
        self._model = AutoModelForImageClassification.from_pretrained("chriamue/bird-species-classifier")

    def classify(self, context: ClassificationCtx) -> dict[str, list[Classification]] | None:
        """Since YOLO does object detection/localization 
        we do not need to do it here before outputting our predictions
        """
        clsf_otp = {}
        if not context.localizations:
            return None
        for lclz in context.localizations:
            inputs = self._extractor(images=context.image[lclz.bbox.ymin:lclz.bbox.ymax,
                                                           lclz.bbox.xmin:
                                                           lclz.bbox.xmax],
                                                            return_tensors="pt")
            with torch.no_grad():
                outputs = self._model(**inputs)
                probabilities = torch.nn.functional.softmax(outputs.logits)
            pred = probabilities.argmax().item()
            pred_conf = probabilities.max().item()
            pred_cls = self._model.config.id2label[pred]
            clsf_otp.setdefault(pred_cls, []).append(Classification(
                                classification=pred_cls,
                                localization=lclz,
                                confidence=float(pred_conf),
                            ))
        return clsf_otp
