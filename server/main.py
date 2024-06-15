import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File
from typing import List

app = FastAPI()


def dummy_neuro_processing(image: np.ndarray) -> np.ndarray:
    """
    Simulates the neuro module's image processing. Here, it simply inverts the image colors.

    :param image: The original image as a numpy array.
    :return: Processed image with inverted colors.
    """
    inverted_image = cv2.bitwise_not(image)
    return inverted_image


@app.post("/process-image/")
async def process_image(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    processed_img = dummy_neuro_processing(img)

    _, encoded_img = cv2.imencode('.png', processed_img)
    return {"image": encoded_img.tobytes()}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
