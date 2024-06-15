from typing import Optional, Dict, Tuple

import cv2
import numpy as np
import streamlit as st
from roboflow.models.object_detection import ObjectDetectionModel
from supervision.detection.core import Detections

from neuro import init_model, RoboflowConfig, predict, ModelConfig, draw_bounding_boxes


@st.cache_resource
def get_model() -> ObjectDetectionModel:
    return init_model(RoboflowConfig())


class Resizer:
    def __init__(self, size: Optional[tuple] = (640, 640)):
        self.size = size
        self.image_size = None

    def resize(self, image: np.ndarray) -> np.ndarray:
        self.image_size = image.shape[:2]
        return cv2.resize(image, self.size)

    def revert(self, image: np.ndarray, predictions: Detections) -> Dict[str, np.array]:
        reverted_image = cv2.resize(image, (self.image_size[1], self.image_size[0]))
        coords = predictions.xyxy
        coords[:, ::2] *= self.image_size[1] / self.size[0]
        coords[:, 1::2] *= self.image_size[0] / self.size[1]
        predictions.xyxy = coords
        return {"image": reverted_image, "predictions": predictions}


class Predictor:
    def __init__(self, model_config: ModelConfig):
        self.model_config = model_config
        self.model = get_model()

    def __call__(self, image: np.ndarray) -> Tuple[Detections, list]:
        predictions = predict(self.model, image, self.model_config)
        return predictions


class Processor:
    def __init__(self):
        self.predictor = Predictor(model_config=ModelConfig())
        self.resizer = Resizer(size=(640, 640))

    def __call__(self, image: np.ndarray) -> np.ndarray:
        resized_image = self.resizer.resize(image)
        predictions, labels = self.predictor(resized_image)
        reverted = self.resizer.revert(image, predictions)
        reverted_image = reverted["image"]
        reverted_predictions = reverted["predictions"]
        annotated_image = draw_bounding_boxes(
            image=reverted_image, detections=reverted_predictions, labels=labels
        )
        return annotated_image
