import glob
import numpy as np
import os
import os
import pandas as pd
import shutil
import torch
from PIL import Image
from super_gradients.training import dataloaders
from super_gradients.training import models
from super_gradients.training.dataloaders.dataloaders import (
    coco_detection_yolo_format_train,
    coco_detection_yolo_format_val
)

from config import CONFIG


def prepare_dataset_format(images_dir: str):
    save_dir = "submits_dir"
    image_paths = glob.glob(os.path.join(images_dir, "*.jpg"))
    for image_path in image_paths:
        image_name = image_path.split('.')[0].split('/')[-1]
        new_image_path = os.path.join(save_dir, image_name, "images", 'val')
        label_path = os.path.join(save_dir, image_name, "labels", 'val')
        os.makedirs(new_image_path, exist_ok=True)
        os.makedirs(label_path, exist_ok=True)
        with open(os.path.join(label_path, f"{image_name}.txt"), "w") as f:
            f.write("0 0.5 0.5 0.2 0.2")
        shutil.copy(image_path, new_image_path)


prepare_dataset_format(CONFIG.path_to_images)


def convert_to_yolo_format(xmin, ymin, xmax, ymax):
    # Вычисляем центр прямоугольника
    x_center = (xmin + xmax) / 2.0
    y_center = (ymin + ymax) / 2.0

    # Вычисляем ширину и высоту прямоугольника
    width = xmax - xmin
    height = ymax - ymin

    return [x_center, y_center, width, height]


model = models.get(
    model_name=CONFIG.model_type,
    num_classes=5,
    checkpoint_path=CONFIG.model_path, )

model = model.eval()

coeff_y = 1.77

BATCH_SIZE = 1
WORKERS = 0
res = pd.DataFrame(
    columns=["filename", "class_id", "rel_x", "rel_y", "width", "height"]
)

for val_dataset_dir in os.listdir('submits_dir'):

    ROOT_DIR = f'submits_dir/{val_dataset_dir}'
    val_imgs_dir = 'images/val'
    val_labels_dir = 'labels/val'
    classes = ['adj', 'int', 'geo', 'pro', 'non']

    dataset_params = {
        'data_dir': ROOT_DIR,
        'val_images_dir': val_imgs_dir,
        'val_labels_dir': val_labels_dir,
        'classes': classes
    }

    val_data = coco_detection_yolo_format_val(
        dataset_params={
            'data_dir': dataset_params['data_dir'],
            'images_dir': dataset_params['val_images_dir'],
            'labels_dir': dataset_params['val_labels_dir'],
            'classes': dataset_params['classes']
        },
        dataloader_params={
            'batch_size': BATCH_SIZE,
            'num_workers': WORKERS
        }
    )

    print(val_dataset_dir)

    path = f'submits_dir/{val_dataset_dir}/images/val/'
    image = Image.open(os.path.join(path, os.listdir(path)[0]))
    print(np.array(image).shape)

    new_height, new_width, _ = np.array(image).shape
    scale_x = new_width / 640
    scale_y = new_height / 640

    for data in val_data:

        image = data[0]
        metadata = data[1]

        pred = model.predict(image, conf=0.5)
        # print(pred)
        labels = pred.prediction.labels.astype(int)
        print('label', labels)
        bbox = list(pred.prediction.bboxes_xyxy)
        # print(bbox)
        for bb, label in zip(bbox, labels):
            bb[0] = bb[0] * scale_x
            bb[1] = bb[1] * scale_y
            bb[2] = bb[2] * scale_x
            bb[3] = bb[3] * scale_y

            bbox_yolov = convert_to_yolo_format(*bb)
            bbox_yolov[0] = bbox_yolov[0] / new_width
            bbox_yolov[1] = bbox_yolov[1] / new_height
            bbox_yolov[2] = bbox_yolov[2] / new_width
            bbox_yolov[3] = bbox_yolov[3] / new_height
            # print(bbox_yolov)

            res = pd.concat(
                (
                    res,
                    pd.DataFrame(
                        {
                            "filename": f'{val_dataset_dir}.jpg',
                            "class_id": label,
                            "rel_x": bbox_yolov[0],
                            "rel_y": bbox_yolov[1] * coeff_y,
                            "width": bbox_yolov[2],
                            "height": bbox_yolov[3] * coeff_y,
                        },
                        index=[0],
                    ),
                ),
                ignore_index=True,
            )

        # print('@###################################################################################')
        # print()

res.to_csv(f"submits/{CONFIG.submit_file_name}", sep=";", index=False)
