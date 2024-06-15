import cv2
import numpy as np
import streamlit as st
import requests


def upload_image_page():
    st.header("Загрузить изображение")
    uploaded_file = st.file_uploader("Загрузите изображение", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        original_image = cv2.imdecode(file_bytes, 1)

        try:
            # Отправляем изображение на сервер для обработки
            files = {'file': uploaded_file.getvalue()}
            response = requests.post("http://localhost:8000/process-image/", files=files)

            if response.status_code == 200:
                processed_image = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)

                # Отображаем оригинальное и обработанное изображение рядом
                col1, col2 = st.columns(2)

                with col1:
                    st.header("Оригинальное изображение")
                    st.image(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB), use_column_width=True)

                with col2:
                    st.header("Обработанное изображение")
                    st.image(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB), use_column_width=True)

                # Блок описания
                st.markdown("### Описание результатов")
                with st.expander("Показать/Скрыть детали"):
                    st.write(
                        "Изображение обработано с использованием модуля dummy neuro. Здесь отображаются результаты.")
            else:
                st.error("Ошибка при обработке изображения.")
        except Exception as e:
            st.error(f"Ошибка: {e}")
