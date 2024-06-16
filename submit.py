import os

import pandas as pd
from imageio.v3 import imread

from app.web.utils.predict import Processor


def convert_xyxy_to_yolov(x_min, y_min, x_max, y_max, img_width, img_height):
    """
    Преобразует координаты разметки из формата xyxy в формат YOLOv.

    Параметры:
    x_min (float): Координата x верхнего левого угла.
    y_min (float): Координата y верхнего левого угла.
    x_max (float): Координата x нижнего правого угла.
    y_max (float): Координата y нижнего правого угла.
    img_width (int): Ширина изображения.
    img_height (int): Высота изображения.

    Возвращает:
    tuple: Координаты в формате YOLOv (x_center, y_center, width, height), нормализованные относительно размеров изображения.
    """
    # Вычисляем центр прямоугольника
    x_center = (x_min + x_max) / 2.0
    y_center = (y_min + y_max) / 2.0

    # Вычисляем ширину и высоту прямоугольника
    width = x_max - x_min
    height = y_max - y_min

    # Нормализуем координаты относительно размеров изображения
    x_center /= img_width
    y_center /= img_height
    width /= img_width
    height /= img_height

    return (x_center, y_center, width, height)


processor = Processor(model_type="yolo_nas")

res = pd.DataFrame(
    columns=["filename", "class_id", "rel_x", "rel_y", "width", "height"]
)
map = {"adj": 0, "int": 1, "geo": 2, "pro": 3, "non": 4}
for file in os.listdir("data/Тестовый датасет"):
    image = imread(os.path.join("data/Тестовый датасет", file))
    x_shape, y_shape, _ = image.shape
    predictions = processor(image)
    entity = []
    print(file)
    for i in range(len(predictions.labels)):
        coordinates = predictions.detections.xyxy

        yolo_coords = convert_xyxy_to_yolov(
            coordinates[i][0],
            coordinates[i][1],
            coordinates[i][2],
            coordinates[i][3],
            x_shape,
            y_shape,
        )
        print(predictions.labels[i])
        res = pd.concat(
            (
                res,
                pd.DataFrame(
                    {
                        "filename": file,
                        "class_id": predictions.labels[i],
                        "rel_x": yolo_coords[0],
                        "rel_y": yolo_coords[1],
                        "width": yolo_coords[2],
                        "height": yolo_coords[3],
                    },
                    index=[0],
                ),
            ),
            ignore_index=True,
        )

res.to_csv("submission.csv", sep=";", index=False)
