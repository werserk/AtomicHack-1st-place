import time

import cv2
import numpy as np
import streamlit as st
from utils import Processor


def upload_image_page():
    st.header("Загрузить изображение")
    uploaded_file = st.file_uploader(
        "Загрузите изображение", type=["jpg", "jpeg", "png"]
    )

    processor = Processor()

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        original_image = cv2.imdecode(file_bytes, 1)

        # try:
        start_time = time.time()
        processed_image = processor(original_image)
        print(f"Time: {time.time() - start_time}")

        # Отображаем оригинальное и обработанное изображение рядом
        col1, col2 = st.columns(2)

        with col1:
            st.header("Оригинальное изображение")
            st.image(
                cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB),
                use_column_width=True,
            )

        with col2:
            st.header("Обработанное изображение")
            st.image(
                cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB),
                use_column_width=True,
            )

        # Блок описания
        st.markdown("### Описание результатов")
        with st.expander("Показать/Скрыть детали"):
            st.write(
                "Изображение обработано с использованием модуля dummy neuro. Здесь отображаются результаты."
            )
        # except Exception as e:
        #     st.error(f"Что-то пошло не так... Пожалуйста, попробуйте ещё раз или другой файл!")
        #     st.error(e)
