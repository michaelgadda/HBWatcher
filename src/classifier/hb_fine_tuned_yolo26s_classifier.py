from src.classifier.classifier import Classifier
from src.data_classes import Classification, ClassificationCtx, \
      Localization, LocType, BBox, ProtocolCtx
from ultralytics import YOLO
from pathlib import Path

class HbFtYolo26sClassifier(Classifier):
    def __init__(self, protocol_params: ProtocolCtx):
        super().__init__(protocol_params)
        base_dir = Path(__file__).resolve().parent.parent.parent
        self._model = YOLO(base_dir / "cache/models--Ultralytics--YOLO26/snapshots/070ac3c51435984d992ba92eb3f234daf4e4500d/yolo26s_20261123_004422.pt")

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
