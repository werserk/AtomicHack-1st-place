import cv2
import numpy as np

from app.neuro.yolo_nas_net.core import YoloNasPredictions

colors = [
    "#5D99E3",
    "#6EE35D",
    "#E35DB9",
    "#E3B45D",
    "#8E6A82",
]

label2color = {
    "adj": colors[0],
    "int": colors[1],
    "geo": colors[2],
    "pro": colors[3],
    "non": colors[4],
}


def hex2rgb(hex: str) -> tuple:
    return tuple(int(hex.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))


def draw_annotations(image: np.ndarray, xyxy: np.ndarray, label: str) -> np.ndarray:
    x1, y1, x2, y2 = xyxy
    color = hex2rgb(label2color[label])

    # plot bounding box
    image = cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)

    # plot label
    image = cv2.putText(image, label, (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
    return image


def plot_yolo_nas_predictions(image: np.ndarray, predictions: YoloNasPredictions) -> np.ndarray:
    length = len(predictions.labels)
    for i in range(length):
        image = draw_annotations(image, predictions.xyxy[i], predictions.labels[i])
    return image


class YoloNasVisualizer:

    @staticmethod
    def plot_predictions(
            image: np.ndarray,
            predictions: YoloNasPredictions,
    ) -> np.ndarray:
        annotated_image = plot_yolo_nas_predictions(image, predictions)
        return annotated_image
