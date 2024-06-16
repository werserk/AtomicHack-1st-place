from io import BytesIO

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse

from app.neuro.roboflow_net import RoboflowModel
from app.neuro.utils import Visualizer
from app.neuro.yolo_nas_net import YoloNasModel
from app.server.utils import decode_image, encode_image

app = FastAPI()
roboflow_model = RoboflowModel()
yolo_nas_model = YoloNasModel()
visualizer = Visualizer()


@app.post("/predict_by_roboflow")
async def predict_by_roboflow(image_file: UploadFile = File(...)):
    content = await image_file.read()
    image = decode_image(content)
    predictions = roboflow_model(image)
    annotated_image = visualizer.plot_predictions(image, predictions)
    bytes_image = encode_image(annotated_image)
    return StreamingResponse(BytesIO(bytes_image), media_type="image/png")


@app.post("/predict_by_yolo_nas")
async def predict_by_yolo_nas(image_file: UploadFile = File(...)):
    content = await image_file.read()
    image = decode_image(content)
    predictions = roboflow_model(image)
    annotated_image = visualizer.plot_predictions(image, predictions)
    bytes_image = encode_image(annotated_image)
    return StreamingResponse(BytesIO(bytes_image), media_type="image/png")
