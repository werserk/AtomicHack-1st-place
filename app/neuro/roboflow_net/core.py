from typing import Optional

import numpy as np
import supervision as sv
from roboflow import Roboflow
from roboflow.models.object_detection import ObjectDetectionModel
from supervision.detection.core import Detections

from app.neuro.roboflow_net.config import RoboflowConfig, ModelConfig


class RoboflowPredictions:
    def __init__(self, detections: Detections, labels: list) -> None:
        self.detections = detections
        self.labels = labels


class RoboflowModel:
    def __init__(
        self,
        roboflow_config: Optional[RoboflowConfig] = None,
        model_config: Optional[ModelConfig] = None,
    ) -> None:
        self.roboflow_config = (
            roboflow_config if roboflow_config is not None else RoboflowConfig()
        )
        self.model_config = model_config if model_config is not None else ModelConfig()
        self.model = self.init_model(self.roboflow_config)

    @staticmethod
    def init_model(config: RoboflowConfig) -> ObjectDetectionModel:
        rf = Roboflow(api_key=config.API_KEY)
        project = rf.workspace().project(config.PROJECT_NAME)
        model = project.version(config.MODEL_VERSION).model
        return model

    def predict(self, image: np.ndarray) -> RoboflowPredictions:
        result = self.model.predict(
            image,
            confidence=self.model_config.confidence,
            overlap=self.model_config.overlap,
        )
        json_result = result.json()
        labels = [item["class"] for item in json_result["predictions"]]
        detections = sv.Detections.from_inference(json_result)
        return RoboflowPredictions(detections=detections, labels=labels)

    def __call__(self, image: np.ndarray) -> RoboflowPredictions:
        return self.predict(image)


class RoboflowVisualizer:
    def __init__(self):
        self.label_annotator = sv.LabelAnnotator()
        self.bounding_box_annotator = sv.BoundingBoxAnnotator()

    def plot_predictions(
        self,
        image: np.ndarray,
        predictions: RoboflowPredictions,
        snow_labels: Optional[bool] = True,
    ) -> np.ndarray:
        detections = predictions.detections
        annotated_image = self.bounding_box_annotator.annotate(
            scene=image, detections=detections
        )

        if snow_labels:
            labels = predictions.labels
            annotated_image = self.label_annotator.annotate(
                scene=annotated_image, detections=detections, labels=labels
            )

        return annotated_image
