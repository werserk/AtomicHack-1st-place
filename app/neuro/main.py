from typing import Tuple

import numpy as np
import supervision as sv
from neuro.config import RoboflowConfig, ModelConfig
from roboflow import Roboflow
from roboflow.models.object_detection import ObjectDetectionModel
from supervision.detection.core import Detections


def init_model(config: RoboflowConfig) -> ObjectDetectionModel:
    rf = Roboflow(api_key=config.API_KEY)
    project = rf.workspace().project(config.PROJECT_NAME)
    model = project.version(config.MODEL_VERSION).model
    return model


def predict(
    model: ObjectDetectionModel, image: np.ndarray, config: ModelConfig
) -> Tuple[Detections, list]:
    result = model.predict(
        image, confidence=config.confidence, overlap=config.overlap
    ).json()
    labels = [item["class"] for item in result["predictions"]]
    detections = sv.Detections.from_inference(result)
    return detections, labels


def draw_bounding_boxes(
    image: np.ndarray, detections: Detections, labels: list
) -> np.ndarray:
    label_annotator = sv.LabelAnnotator()
    bounding_box_annotator = sv.BoundingBoxAnnotator()

    annotated_image = bounding_box_annotator.annotate(
        scene=image, detections=detections
    )
    annotated_image = label_annotator.annotate(
        scene=annotated_image, detections=detections, labels=labels
    )

    return annotated_image


def main():
    import cv2

    model = init_model(config=RoboflowConfig())
    image_path = "../../data/test.jpg"
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    detections, labels = predict(model=model, image=image, config=ModelConfig())
    annotated_image = draw_bounding_boxes(
        image=cv2.imread(image_path), detections=detections, labels=labels
    )
    cv2.imwrite(
        "../../data/output.jpg", cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
    )


if __name__ == "__main__":
    main()
