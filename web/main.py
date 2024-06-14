import streamlit as st

from st_pages import capture_video, upload_image, welcome


def main():
    st.set_page_config(
        page_title="Defect Detection App", page_icon=":camera:")

    page_names_to_funcs = {
        "Приветственная страница": welcome.welcome_page,
        "Обработка с камеры": capture_video.capture_video_page,
        "Обработка изображений": upload_image.upload_image_page,
    }

    demo_name = st.sidebar.selectbox("Выберите страницу", page_names_to_funcs.keys())
    page_names_to_funcs[demo_name]()


if __name__ == "__main__":
    main()
