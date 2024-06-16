import datetime
import os
from typing import Optional

import cv2
import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile

from app.web.utils import get_processor
from app.web.utils.predict import Processor

TEMP_DIR = "data/temp"
os.makedirs(TEMP_DIR, exist_ok=True)


def upload_video_page() -> None:
    st.header("Загрузить видео")
    uploaded_file = st.file_uploader("Загрузите видео", type=["mp4", "avi", "mov"])

    processor = get_processor()

    if uploaded_file is not None:
        original_video = load_video(uploaded_file)
        if original_video is not None:
            display_video(original_video, processor)


def load_video(uploaded_file: UploadedFile) -> Optional[str]:
    try:
        file_bytes = uploaded_file.read()
        current_datetime = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        file_path = os.path.join(TEMP_DIR, f"temp_video_{current_datetime}.{uploaded_file.name.split('.')[-1]}")
        with open(file_path, "wb") as f:
            f.write(file_bytes)
        return file_path
    except Exception as e:
        st.error("Не удалось загрузить видео. Пожалуйста, попробуйте ещё раз.")
        st.error(e)
        return None


def display_video(video_path: str, processor: Processor, delete_temp_video: Optional[bool] = True) -> None:
    # try:
        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        original_fps = int(cap.get(cv2.CAP_PROP_FPS))

        print(f"Video resolution: {width}x{height}")
        print(f"Original FPS: {original_fps}")

        target_fps = 5
        frame_interval = max(1, original_fps // target_fps)
        print(f"Frame interval: {frame_interval}")

        processed_video_path = os.path.join(TEMP_DIR, f"processed_{video_path}")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(processed_video_path, fourcc, target_fps, (width, height))

        total_frames_to_process = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) // frame_interval
        print(f"Total frames to process: {total_frames_to_process}")

        frame_count = 0
        frame_processed = 0

        progress_bar = st.progress(0, text="Обработка видео...")
        prev_frame = None

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                predictions = processor(frame)
                annotated_frame = processor.annotate_image(frame, predictions)
                out.write(annotated_frame)
                frame_processed += 1
                progress_bar.progress(min(frame_processed / total_frames_to_process, 1), text="Обработка видео...")
                prev_frame = annotated_frame
            else:
                if prev_frame is None:
                    continue
                out.write(prev_frame)

            frame_count += 1

        cap.release()
        out.release()

        progress_bar.empty()

        display_results_description(processed_video_path)
        current_datetime = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        st.download_button(
            label="📥 Скачать видео",
            data=processed_video_path,
            file_name=f"processed_video_{current_datetime}.mp4",
        )

        if delete_temp_video and os.path.exists(video_path):
            os.remove(video_path)

    # except Exception as e:
    #     st.warning("Что-то пошло не так... Пожалуйста, попробуйте ещё раз или другой файл!")
    #     st.error(e)


def display_results_description(processed_video_path: str) -> None:
    st.header("Обработанное видео")
    st.video(processed_video_path)  # TODO: fix video display

    st.download_button(
        label="📥 Скачать видео",
        data=processed_video_path,
        file_name=processed_video_path,
    )


if __name__ == "__main__":
    upload_video_page()
