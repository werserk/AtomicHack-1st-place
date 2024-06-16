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

    if st.button("Обработать видео") and uploaded_file is not None:
        original_video = load_video(uploaded_file)
        if original_video is not None:
            display_video(original_video, processor)


def load_video(uploaded_file: UploadedFile) -> Optional[str]:
    try:
        file_bytes = uploaded_file.read()
        current_datetime = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        file_path = os.path.join(
            TEMP_DIR,
            f"temp_video_{current_datetime}.{uploaded_file.name.split('.')[-1]}",
        )
        with open(file_path, "wb") as f:
            f.write(file_bytes)
        return file_path
    except Exception as e:
        st.error("Не удалось загрузить видео. Пожалуйста, попробуйте ещё раз.")
        st.error(e)
        return None


def display_video(video_path: str, processor: Processor) -> None:
    try:
        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        original_fps = int(cap.get(cv2.CAP_PROP_FPS))

        target_fps = 5
        frame_interval = max(1, original_fps // target_fps)

        processed_video_path = video_path.replace(".mp4", "_processed.mp4")
        video_writer = cv2.VideoWriter(
            processed_video_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            target_fps,
            (width, height),
        )

        total_frames_to_process = (
            int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) // frame_interval
        )

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
                video_writer.write(annotated_frame)
                frame_processed += 1
                progress_bar.progress(
                    min(frame_processed / total_frames_to_process, 1),
                    text="Обработка видео...",
                )
                prev_frame = annotated_frame
            else:
                if prev_frame is None:
                    continue
                video_writer.write(prev_frame)

            frame_count += 1

        cap.release()
        video_writer.release()

        progress_bar.empty()

        display_results_description(processed_video_path)

    except Exception as e:
        st.warning(
            "Что-то пошло не так... Пожалуйста, попробуйте ещё раз или другой файл!"
        )
        st.error(e)


def display_results_description(processed_video_path: str) -> None:
    st.header("Обработанное видео")

    # video_file = open(processed_video_path, "rb")
    # video_bytes = video_file.read()
    # st.video(video_bytes)

    current_datetime = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    if os.path.exists(processed_video_path):
        with open(processed_video_path, "rb") as file:
            st.download_button(
                label="📥 Скачать видео",
                data=file,
                file_name=f"processed_video_{current_datetime}.mp4",
                mime="video/mpeg",
            )
    else:
        st.error("Процессированное видео не найдено.")


if __name__ == "__main__":
    upload_video_page()
