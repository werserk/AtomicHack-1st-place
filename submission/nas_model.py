from super_gradients.training import models
from super_gradients.common.object_names import Models
import torch
import numpy as np

class NasModel:
    def __init__(self, weights_path: str):
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.model = models.get(model_name='yolo_nas_s', num_classes=5, checkpoint_path=weights_path).to(device)

    def predict(self, image):
        preds = self.model.predict(image)

        bboxes_xyxy = preds.prediction.bboxes_xyxy
        labels = preds.prediction.labels
        conf = preds.prediction.confidence

        return bboxes_xyxy, labels, conf

