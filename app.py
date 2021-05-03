from util import detect_license_plate
from PIL import Image
import streamlit as st
st.title("License Plate Recognition")

if __name__ == "__main__":
    uploaded_file = st.file_uploader("Choose the image", type="jpg")
    if(uploaded_file is not None):
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        license_plate_character = detect_license_plate.run(image)
        if((license_plate_character is None) or (license_plate_character == "")):
            st.header("No characters found")
        else:
            st.header(license_plate_character)
    else:
        st.write("")