from src.classifier.classifier import Classifier
from src.data_classes import Classification, ClassificationCtx, Localization, LocType, BBox
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

class Yolo26sClassifier(Classifier):
    def __init__(self):
        self._weights = hf_hub_download("Ultralytics/YOLO26", "yolo26s.pt", cache_dir="./cache")
        self._model = YOLO(self._weights)

    def classify(self, context: ClassificationCtx) -> list[Classification]:
        """Since YOLO does object detection/localization 
        we do not need to do it here before outputting our predictions
        """
        preds = self._model.predict(context.image)
        class_dict = preds[0].names
        clsf_otp = []
        if preds[0].boxes is not None:
            box_classes = preds[0].boxes.cls
            for index, box in enumerate(preds[0].boxes.xyxy.cpu().numpy()):
                x1, y1, x2, y2 = map(int, box)
                clsf_otp.append(Classification(class_dict[box_classes[index].item()], 
                                               Localization(LocType.BoundingBox, 
                                                            BBox(x1, y1, x2, y2))))
        return clsf_otp