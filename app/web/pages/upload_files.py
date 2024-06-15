import cv2
import numpy as np
import streamlit as st

from app.web.utils import get_processor


def upload_files_page():
    st.header("Загрузить изображение")
    uploaded_file = st.file_uploader(
        "Загрузите изображение", type=["jpg", "jpeg", "png"]
    )

    processor = get_processor()

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        original_image = cv2.imdecode(file_bytes, 1)

        try:
            processed_image = processor(original_image)

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

            st.markdown("### Описание результатов")
            with st.expander("Показать/Скрыть детали"):
                st.write(
                    "Изображение обработано с использованием модуля dummy neuro. Здесь отображаются результаты."
                )
        except Exception as e:
            st.error(
                f"Что-то пошло не так... Пожалуйста, попробуйте ещё раз или другой файл!"
            )
            st.error(e)
