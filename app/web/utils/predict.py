import logging
from typing import Optional, Dict

import cv2
import numpy as np
import streamlit as st
from supervision.detection.core import Detections

from app.neuro.roboflow_net import RoboflowModel
from app.neuro.utils import Predictions, Visualizer
from app.neuro.yolo_nas_net import YoloNasModel

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Resizer:
    def __init__(self, size: Optional[tuple] = (640, 640)):
        self.size = size
        self.image_size = None

    def apply(self, image: np.ndarray) -> np.ndarray:
        self.image_size = image.shape[:2]
        return cv2.resize(image, self.size)

    def revert(self, image: np.ndarray, detections: Detections) -> Dict[str, np.array]:
        reverted_image = cv2.resize(image, (self.image_size[1], self.image_size[0]))
        coords = detections.xyxy
        coords[:, ::2] *= self.image_size[1] / self.size[0]
        coords[:, 1::2] *= self.image_size[0] / self.size[1]
        detections.xyxy = coords
        return {"image": reverted_image, "detections": detections}


class Processor:
    def __init__(self, model_type: str) -> None:
        if model_type == "roboflow":
            self.predictor = RoboflowModel()
        elif model_type == "yolo_nas":
            self.predictor = YoloNasModel()
        else:
            raise ValueError(
                f"Unknown model type: {model_type}. We only support `roboflow` and `yolo_nas`."
            )
        self.resizer = Resizer()
        self.visualizer = Visualizer()

    def predict_image(self, image: np.ndarray) -> np.ndarray:
        resized_image = self.resizer.apply(image)
        predictions = self.predictor(resized_image)
        reverted = self.resizer.revert(image, predictions.detections)
        reverted_image = reverted["image"]
        predictions.detections = reverted["detections"]
        annotated_image = self.visualizer.plot_predictions(
            image=reverted_image, predictions=predictions
        )
        return annotated_image

    def predict(self, image: np.ndarray) -> Predictions:
        resized_image = self.resizer.apply(image)
        predictions = self.predictor(resized_image)
        return predictions

    def annotate_image(self, image: np.ndarray, predictions: Predictions) -> np.ndarray:
        reverted = self.resizer.revert(image, predictions.detections)
        reverted_image = reverted["image"]
        predictions.detections = reverted["detections"]
        annotated_image = self.visualizer.plot_predictions(
            image=reverted_image, predictions=predictions
        )
        return annotated_image

    def __call__(self, image: np.ndarray) -> Predictions:
        return self.predict(image)


@st.cache_resource
def get_processor() -> Processor:
    return Processor(model_type="yolo_nas")
