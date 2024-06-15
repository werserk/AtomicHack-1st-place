import av
import cv2
import numpy as np
import requests
import streamlit as st
from streamlit_webrtc import webrtc_streamer


def process_frame_via_api(image: np.ndarray) -> np.ndarray:
    """
    Sends the frame to the API for processing.

    :param image: The original frame as a numpy array.
    :return: Processed frame with inverted colors.
    """
    _, encoded_img = cv2.imencode('.png', image)
    files = {'file': encoded_img.tobytes()}
    response = requests.post("http://localhost:8000/process-image/", files=files)

    if response.status_code == 200:
        processed_image = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)
        return processed_image
    else:
        raise Exception("Error processing frame via API")


def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    image = frame.to_ndarray(format="bgr24")
    try:
        processed_image = process_frame_via_api(image)
    except Exception as e:
        st.error(f"Error: {e}")
        return frame
    return av.VideoFrame.from_ndarray(processed_image, format="bgr24")


def capture_video_page():
    st.header("Съёмка видео")
    st.markdown("Используйте камеру для захвата видео. Видео будет обработано в реальном времени.")

    ctx = webrtc_streamer(
        key="defect-detection",
        video_frame_callback=video_frame_callback,
        media_stream_constraints={"video": True, "audio": False},
    )
