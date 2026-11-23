from src.classifier.classifier import Classifier
from src.data_classes import Classification, ClassificationCtx, \
      Localization, LocType, BBox, ProtocolCtx

from transformers import AutoFeatureExtractor, AutoModelForImageClassification


class ChriamueBirdClassifier(Classifier):
    def __init__(self, protocol_params: ProtocolCtx):
        super().__init__(protocol_params)
        extractor = AutoFeatureExtractor.from_pretrained("chriamue/bird-species-classifier")
        self._model = AutoModelForImageClassification.from_pretrained("chriamue/bird-species-classifier")

    def classify(self, context: ClassificationCtx) -> dict[str, list[Classification]]:
        """Since YOLO does object detection/localization 
        we do not need to do it here before outputting our predictions
        """
        preds = self._model.predict(context.image, iou=self.protocol_params.iou, conf=self.protocol_params.conf)
        class_dict = preds[0].names
        clsf_otp = {}
        if preds[0].boxes is not None:
            box_classes = preds[0].boxes.cls
            box_confidences = preds[0].boxes.conf
            for index, box in enumerate(preds[0].boxes.xyxy.cpu().numpy()):
                x1, y1, x2, y2 = map(int, box)
                label = class_dict[box_classes[index].item()]
                clsf_otp.setdefault(label, []).append(Classification(
                    classification=label,
                    localization=Localization(LocType.BoundingBox, BBox(x1, y1, x2, y2)),
                    confidence=float(box_confidences[index].item()),
                ))
        return clsf_otp
