import logging

import av
import cv2
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode

from app.web.utils import get_processor

logger = logging.getLogger(__name__)


def dummy_callback(frame: av.VideoFrame) -> av.VideoFrame:
    image = frame.to_ndarray(format="bgr24")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edged = cv2.Canny(gray, 30, 200)
    contours, hierarchy = cv2.findContours(
        edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    annotated_image = cv2.drawContours(image, contours, -1, (0, 255, 0), 1)
    return av.VideoFrame.from_ndarray(annotated_image, format="bgr24")


class VideoCallback:
    def __init__(self):
        self.processor = get_processor()

    def __call__(self, frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray(format="bgr24")
        predictions = self.processor(image)
        annotated_image = self.processor.annotate_image(image, predictions)
        return av.VideoFrame.from_ndarray(annotated_image, format="bgr24")


def capture_video_page():
    st.header("Съёмка видео")
    st.markdown(
        "Используйте камеру для захвата видео. Видео будет обработано в реальном времени."
    )
    st.markdown(
        "Наш сервер не имеет достаточной вычислительной мощности, чтобы поддерживать инференс модели. "
        "Однако мы готовы продемонстрировать технологию в реальной среде, поэтому представили модуль с камерой."
    )

    video_frame_callback = dummy_callback

    ctx = webrtc_streamer(
        key="defect-detection",
        # If you have troubles with ICE (bad network connection) swap to dummy_callback
        video_frame_callback=video_frame_callback,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=False,
        mode=WebRtcMode.SENDRECV,
    )
