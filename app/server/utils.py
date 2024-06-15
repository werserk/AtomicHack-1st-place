import cv2
import numpy as np


def decode_image(content: bytes) -> np.ndarray:
    image = np.frombuffer(content, np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image


def encode_image(image: np.ndarray) -> bytes:
    _, encoded_image = cv2.imencode(".png", image)
    return encoded_image.tobytes()
