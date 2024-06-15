from io import BytesIO

import uvicorn
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse

from app.neuro.roboflow_net import RoboflowModel, RoboflowVisualizer
from app.server.config import UvicornConfig
from app.server.utils import decode_image, encode_image

app = FastAPI()
roboflow_model = RoboflowModel()
roboflow_visualizer = RoboflowVisualizer()


@app.post("/predict_by_roboflow")
async def predict_by_roboflow(file: UploadFile = File(...)):
    content = await file.read()
    image = decode_image(content)
    predictions = roboflow_model(image)
    annotated_image = roboflow_visualizer.plot_predictions(image, predictions)
    bytes_image = encode_image(annotated_image)
    return StreamingResponse(BytesIO(bytes_image), media_type="image/png")


def start_uvicorn():
    uvicorn.run(app, host=UvicornConfig.HOST, port=UvicornConfig.PORT)
