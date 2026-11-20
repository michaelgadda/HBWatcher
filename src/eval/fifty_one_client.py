import fiftyone as fo
from src.data_classes import ClassificationCtx
import cv2
from src.protocol.protocol import Protocol
from src.classifier.classifier import Classifier
from src.localizer.localizer import Localizer

class FiftyOneClient:
    def __init__(self):
        self.dataset = fo.Dataset()

    def import_coco_gt_annotations(self, coco_json_path: str, data_path: str):
        self.dataset.add_dir( 
            dataset_type=fo.types.COCODetectionDataset,
            data_path=data_path,
            labels_path=coco_json_path,
            label_field="ground_truth")

    def eval_protocol_pred(self, protocol: Protocol, eval_key: str = "v1", iou: float = .5):
        """Save protocol predictions, including confidence, alongside ground truth."""
        for sample in self.dataset:
            image = cv2.imread(sample.filepath)
            if image is None:
                raise ValueError(f"Unable to read image: {sample.filepath}")
            height, width = image.shape[:2]
            preds = protocol.predict(image)

            detections = []
            for _, clsfs in preds.items():
                for pred in clsfs:
                    bbox = pred.localization.bbox
                    if bbox is None:
                        raise ValueError("Detection evaluation requires bounding boxes")
                    detections.append(fo.Detection(
                        label=pred.classification,
                        bounding_box=[
                            bbox.xmin / width,
                            bbox.ymin / height,
                            (bbox.xmax - bbox.xmin) / width,
                            (bbox.ymax - bbox.ymin) / height,
                        ],
                        confidence=pred.confidence,
                    ))

            sample["predictions"] = fo.Detections(detections=detections)
            sample.save()
        results = self.dataset.evaluate_detections("predictions", 
                                                gt_field="ground_truth",
                                                method="coco",
                                                eval_key=eval_key, 
                                                iou=iou,\
                                                classwise=True,
                                                compute_mAP=True)
        results.print_report()
        print("COCO mAP:", results.mAP())



    def eval_clsf_pred(self, classifier: Classifier, eval_key: str = "classification_v1"):
        """Evaluate one class per full image, without comparing bounding boxes.

        Ground truth may be a Classification or COCO Detections containing one
        distinct class. The classifier must return at most one prediction per
        image; absent predictions count as missing rather than being skipped.
        """
        for sample in self.dataset:
            ground_truth = sample["ground_truth"]
            if isinstance(ground_truth, fo.Classification):
                label = ground_truth.label
            elif isinstance(ground_truth, fo.Detections):
                labels = {detection.label for detection in ground_truth.detections}
                if len(labels) != 1:
                    raise ValueError(
                        f"Classification evaluation requires one ground-truth class "
                        f"per image: {sample.filepath}"
                    )
                label = labels.pop()
            else:
                raise ValueError(f"Missing classification ground truth: {sample.filepath}")
            if not label:
                raise ValueError(f"Missing classification ground truth: {sample.filepath}")

            image = cv2.imread(sample.filepath)
            if image is None:
                raise ValueError(f"Unable to read image: {sample.filepath}")
            predictions = classifier.classify(ClassificationCtx(image=image))
            predictions = [pred for clsfs in predictions.values() for pred in clsfs]
            if len(predictions) > 1:
                raise ValueError(
                    f"Classification evaluation requires at most one prediction "
                    f"per image: {sample.filepath}"
                )

            sample["classification_ground_truth"] = fo.Classification(label=label)
            sample["classification_predictions"] = (
                fo.Classification(
                    label=predictions[0].classification,
                    confidence=predictions[0].confidence,
                ) if predictions else fo.Classification()
            )
            sample.save()

        results = self.dataset.evaluate_classifications(
            "classification_predictions",
            gt_field="classification_ground_truth",
            eval_key=eval_key,
        )
        results.print_report()
        return results

    def eval_lclz_pred(
        self, localizer: Localizer, eval_key: str = "localization_v1", iou: float = .5,
    ):
        """Evaluate boxes independently of class labels, at the given IoU."""
        for sample in self.dataset:
            ground_truth = sample["ground_truth"]
            if not isinstance(ground_truth, fo.Detections):
                raise ValueError(f"Localization evaluation requires ground-truth detections: {sample.filepath}")
            # Use a common class on both sides so the report measures geometry.
            # Keep the original species labels and crowd attributes intact.
            localization_gt = ground_truth.copy()
            for detection in localization_gt.detections:
                detection.label = "object"
            image = cv2.imread(sample.filepath)
            if image is None:
                raise ValueError(f"Unable to read image: {sample.filepath}")
            height, width = image.shape[:2]
            detections = []
            for localization in localizer.localize(image):
                bbox = localization.bbox
                if bbox is None:
                    raise ValueError("Localization evaluation requires bounding boxes")
                detections.append(fo.Detection(
                    label="object",
                    bounding_box=[
                        bbox.xmin / width,
                        bbox.ymin / height,
                        (bbox.xmax - bbox.xmin) / width,
                        (bbox.ymax - bbox.ymin) / height,
                    ],
                ))
            sample["localization_ground_truth"] = localization_gt
            sample["localization_predictions"] = fo.Detections(detections=detections)
            sample.save()

        results = self.dataset.evaluate_detections(
            "localization_predictions",
            gt_field="localization_ground_truth",
            method="coco",
            eval_key=eval_key,
            iou=iou,
            classwise=False,
        )
        results.print_report()
        return results

