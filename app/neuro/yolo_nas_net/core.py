from typing import Optional

import numpy as np
import supervision as sv
import torch
from super_gradients.training import models

from app.neuro.utils import Predictions
from app.neuro.yolo_nas_net.config import YoloNasConfig


class YoloNasModel:
    def __init__(self, config: Optional[YoloNasConfig] = None) -> None:
        self.config = config if config is not None else YoloNasConfig()
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.model = models.get(
            model_name=self.config.MODEL_NAME,
            num_classes=self.config.NUM_CLASSES,
            checkpoint_path=self.config.WEIGHTS_PATH,
        ).to(device)

    def predict(self, image: np.ndarray) -> Predictions:
        predictions = self.model.predict(image, conf=0.2, iou=0.5)
        detections = sv.Detections.from_yolo_nas(predictions)
        labels = [self.config.CLASSES[i] for i in predictions.prediction.labels]
        return Predictions(detections=detections, labels=labels)

    def __call__(self, image: np.ndarray) -> Predictions:
        return self.predict(image)
