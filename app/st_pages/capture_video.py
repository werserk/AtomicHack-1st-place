import time

import av
import streamlit as st
from streamlit_webrtc import webrtc_streamer
from utils import Processor


class VideoCallback:
    def __init__(self, processor: Processor):
        self.processor = processor

    def __call__(self, frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray(format="bgr24")

        start_time = time.time()
        processed_image = self.processor(image)
        delta_time = time.time() - start_time
        print(f"FPS: {1 / delta_time: .1f} / Time: {delta_time:.3f}")

        return av.VideoFrame.from_ndarray(processed_image, format="bgr24")


def capture_video_page():
    st.header("Съёмка видео")
    st.markdown(
        "Используйте камеру для захвата видео. Видео будет обработано в реальном времени."
    )

    video_frame_callback = VideoCallback(processor=Processor())

    ctx = webrtc_streamer(
        key="defect-detection",
        video_frame_callback=video_frame_callback,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=False,
    )
