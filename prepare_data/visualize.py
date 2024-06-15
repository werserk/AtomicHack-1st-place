import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches


def add_labels_to_image(image: np.ndarray, defects: dict, path_to_save: str, color_map: dict):
    fig, ax = plt.subplots(1)

    ax.imshow(image)
    y_shape, x_shape, _ = image.shape
    for label, cords in defects.items():
        for cord in cords:
            x_center = cord[0] * x_shape
            y_center = cord[1] * y_shape
            width = cord[2] * x_shape
            height = cord[3] * y_shape

            x_min = int(x_center - width / 2)
            y_min = int(y_center - height / 2)

            rect = patches.Rectangle((x_min, y_min), width, height, linewidth=1, edgecolor=color_map[label],
                                     facecolor='none')
            ax.add_patch(rect)

            # top_left_x = x_center - width / 2
            # top_left_y = y_center - height / 2
            #
            # ax.text(top_left_x, top_left_y + 5, label, color=color_map[label], fontsize=2, ha='left', va='bottom')

    ax.axis('off')
    plt.savefig(path_to_save, bbox_inches='tight', pad_inches=0, dpi=1024)
    plt.close(fig)


def show_defects(image: np.ndarray, defects: dict):
    fig, ax = plt.subplots(1, 2, figsize=(10, 15))
    ax[0].imshow(image)

    ax[1].imshow(image)
    y_shape, x_shape, _ = image.shape
    for label, cords in defects.items():
        for cord in cords:
            x_center = cord[0] * x_shape
            y_center = cord[1] * y_shape
            width = cord[2] * x_shape
            height = cord[3] * y_shape

            x_min = int(x_center - width / 2)
            y_min = int(y_center - height / 2)

            rect = patches.Rectangle((x_min, y_min), width, height, linewidth=1, edgecolor='r', facecolor='none')
            ax[1].add_patch(rect)
