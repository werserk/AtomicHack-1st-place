from typing import Optional

import numpy as np
import torch
from super_gradients.training import models

from app.neuro.yolo_nas_net.config import YoloNasConfig


class YoloNasPredictions:
    def __init__(self, xyxy: np.ndarray, labels: np.ndarray, conf: np.ndarray) -> None:
        self.xyxy = xyxy
        self.labels = labels
        self.conf = conf


class YoloNasModel:
    def __init__(self, config: Optional[YoloNasConfig] = None) -> None:
        self.config = config if config is not None else YoloNasConfig()
        device = (
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.model = models.get(
            model_name=self.config.MODEL_NAME,
            num_classes=self.config.NUM_CLASSES,
            checkpoint_path=self.config.WEIGHTS_PATH,
        ).to(device)

    def predict(self, image: np.ndarray) -> YoloNasPredictions:
        predictions = self.model.predict(image)
        return YoloNasPredictions(
            xyxy=predictions.prediction.bboxes_xyxy,
            labels=predictions.prediction.labels,
            conf=predictions.prediction.confidence,
        )

    def __call__(self, image: np.ndarray) -> YoloNasPredictions:
        return self.predict(image)
