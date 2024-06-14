import av
import streamlit as st
from streamlit_webrtc import webrtc_streamer

from utils import dummy_neuro_processing


def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    image = frame.to_ndarray(format="bgr24")
    image = dummy_neuro_processing(image)
    return av.VideoFrame.from_ndarray(image, format="bgr24")


def capture_video_page():
    st.header("Съёмка видео")
    st.markdown(
        "Используйте камеру для захвата видео. Видео будет обработано в реальном времени."
    )

    ctx = webrtc_streamer(
        key="defect-detection",
        video_frame_callback=video_frame_callback,
        media_stream_constraints={"video": True, "audio": False},
    )
