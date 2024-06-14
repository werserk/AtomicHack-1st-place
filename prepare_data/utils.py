import os
import random
from typing import Tuple
def create_folder(path: str):
    if path.split('/')[-1] not in os.listdir('/'.join(path.split('/')[:-1])):
        os.mkdir(path)

def get_random_image(path: str) -> Tuple[str, str]:
    image_name = random.choice(os.listdir(os.path.join(path, 'images')))
    return os.path.join(path, 'images', image_name), os.path.join(path, 'labels', image_name.replace('jpg', 'txt'))