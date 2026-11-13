import numpy as np
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from src.localizer.localizer import Localizer
from src.data_classes import Localization, LocType, BBox, ProtocolCtx

class Yolo26sLocalizer(Localizer):
    def __init__(self, protocol_params: ProtocolCtx):
        super().__init__(protocol_params)
        self._weights = hf_hub_download("Ultralytics/YOLO26", "yolo26s.pt", cache_Dir="./cache")
        self._model = YOLO(self.weights)

    def weights(self):
        return self._weights

    def model(self):
        return self._model

    def localize(self, image: np.ndarray) -> list[Localization]:
        """Since YOLO does object detection/localization 
        we do not need to do it here before outputting our predictions
        """
        preds = self.model.predict(image, iou=self.protocol_params.iou, conf=self.protocol_params.conf)
        
        localizations = []
        if preds[0].boxes is not None:
            for box in preds[0].boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = map(int, box)
                localizations.append(Localization(LocType.BoundingBox, 
                                                            BBox(x1, y1, x2, y2)))
        return localizations