import streamlit as st

from app.web.pages import capture_video, upload_files, welcome


def start_web_app():
    st.set_page_config(
        page_title="Defect Detection App",
        page_icon="📦",
    )

    page_names_to_funcs = {
        "Приветственная страница": welcome.welcome_page,
        "Обработка с камеры": capture_video.capture_video_page,
        "Обработка файлов": upload_files.upload_files_page,
    }

    demo_name = st.sidebar.selectbox("Выберите страницу", page_names_to_funcs.keys())
    page_names_to_funcs[demo_name]()
