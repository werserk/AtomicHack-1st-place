import cv2
import numpy as np
import streamlit as st


def dummy_neuro_processing(image: np.ndarray) -> np.ndarray:
    """
    Simulates the neuro module's image processing. Here, it simply inverts the image colors.

    :param image: The original image as a numpy array.
    :return: Processed image with inverted colors.
    """
    inverted_image = cv2.bitwise_not(image)
    return inverted_image


def main():
    st.title("Image Processing App with Neuro Module")

    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        original_image = cv2.imdecode(file_bytes, 1)

        # Process the image using the dummy neuro function
        processed_image = dummy_neuro_processing(original_image)

        # Display the original and processed images side by side
        col1, col2 = st.columns(2)

        with col1:
            st.header("Original Image")
            st.image(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB), use_column_width=True)

        with col2:
            st.header("Processed Image")
            st.image(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB), use_column_width=True)

        # Description block
        st.markdown("### Description of Findings")
        with st.expander("Show/Hide Details"):
            st.write("The image is processed using the dummy neuro module. The image is then displayed here.")


if __name__ == "__main__":
    main()
