import datetime
from typing import Optional

import cv2
import numpy as np
import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile

from app.neuro.roboflow_net.core import RoboflowPredictions
from app.web.utils import get_processor
from app.web.utils.predict import Processor
from app.web.utils.tables import generate_table, dataframe_to_excel_bytes


def upload_image_page() -> None:
    st.header("Загрузить изображение")
    uploaded_file = st.file_uploader("Загрузите изображение", type=["jpg", "jpeg", "png"])

    processor = get_processor()

    if uploaded_file is not None:
        original_image = load_image(uploaded_file)
        if original_image is not None:
            display_images(original_image, processor)


def load_image(uploaded_file: UploadedFile) -> Optional[np.array]:
    try:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        return image
    except Exception as e:
        st.error("Не удалось загрузить изображение. Пожалуйста, попробуйте ещё раз.")
        st.error(e)
        return None


def display_images(original_image: np.array, processor: Processor) -> None:
    try:
        predictions = processor(original_image)
        processed_image = processor.annotate_image(original_image, predictions)

        st.header("Обработанное изображение")
        st.image(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))

        display_results_description(predictions)
    except Exception:
        st.warning("Что-то пошло не так... Пожалуйста, попробуйте ещё раз или другой файл!")


def display_results_description(predictions: RoboflowPredictions) -> None:
    st.markdown("### Описание результатов")
    defects_table = generate_table(predictions)

    with st.expander("Показать/Скрыть детализацию дефектов"):
        st.dataframe(defects_table, use_container_width=True)

    current_datetime = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    dataframe_name = f"defects_{current_datetime}.xlsx"

    df_xlsx = dataframe_to_excel_bytes(defects_table)

    st.download_button(
        label="📥 Скачать отчёт",
        data=df_xlsx,
        file_name=dataframe_name,
    )
