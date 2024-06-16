from typing import Optional, List

import numpy as np
import supervision as sv
from supervision.detection.core import Detections


class Predictions:
    def __init__(self, detections: Detections, labels: List[str]) -> None:
        self.detections = detections
        self.labels = labels


def yolo_nas_predictions_to_supervision_detections(
        predictions: np.ndarray,
) -> sv.Detections:
    return sv.Detections.from_yolo_nas(predictions)


class Visualizer:
    def __init__(self):
        self.label_annotator = sv.LabelAnnotator()
        self.bounding_box_annotator = sv.BoundingBoxAnnotator()

    def plot_predictions(
            self,
            image: np.ndarray,
            predictions: Predictions,
            snow_labels: Optional[bool] = True,
    ) -> np.ndarray:
        detections = predictions.detections
        annotated_image = self.bounding_box_annotator.annotate(
            scene=image, detections=detections
        )

        if snow_labels:
            labels = predictions.labels
            labels = [str(label) for label in labels]
            annotated_image = self.label_annotator.annotate(
                scene=annotated_image, detections=detections, labels=labels
            )

        return annotated_image
