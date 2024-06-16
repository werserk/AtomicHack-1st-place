from typing import Optional

import numpy as np
import supervision as sv
from roboflow import Roboflow
from roboflow.models.object_detection import ObjectDetectionModel

from app.neuro.roboflow_net.config import RoboflowConfig, ModelConfig
from app.neuro.utils import Predictions


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

    def predict(self, image: np.ndarray) -> Predictions:
        result = self.model.predict(
            image,
            confidence=self.model_config.confidence,
            overlap=self.model_config.overlap,
        )
        json_result = result.json()
        labels = [item["class"] for item in json_result["predictions"]]
        detections = sv.Detections.from_inference(json_result)
        return Predictions(detections=detections, labels=labels)

    def __call__(self, image: np.ndarray) -> Predictions:
        return self.predict(image)
