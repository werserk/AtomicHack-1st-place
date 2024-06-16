import streamlit as st

from app.web.pages import welcome_page, capture_video_page, upload_image_page, upload_video_page


def start_web_app():
    st.set_page_config(
        page_title="Поиск дефектов сварных швов",
        page_icon="🔍",
    )

    page_names_to_funcs = {
        "О нас": welcome_page,
        "Обработка с камеры": capture_video_page,
        "Обработка изображения": upload_image_page,
        "Обработка видео": upload_video_page
    }

    demo_name = st.sidebar.selectbox("Выберите страницу", page_names_to_funcs.keys())
    page_names_to_funcs[demo_name]()
