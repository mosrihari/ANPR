from utils import wpod_file_read
from PIL import Image
import streamlit as st
from utils.image_preprocessing import preprocess_image
from utils.image_preprocessing import get_plate
from utils import character_segmentation
from model import prediction
st.title("License Plate Recognition")


if __name__ == "__main__":
    wpod_model = wpod_file_read.run()
    uploaded_file = st.file_uploader("Choose the image", type="jpg")
    if(uploaded_file is not None):
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        test_image = preprocess_image(image, resize_bool=True)
        try:
            LpImg, cor = get_plate(test_image, wpod_model)
        except:
            print("No License plate detected")
            LpImg = None
        # Character segmentation
        if(LpImg is not None):
            crop_characters = character_segmentation.run(LpImg)
            # Model for character recognition
            license_plate_character = prediction.run(crop_characters)
            if((license_plate_character is None) or (license_plate_character == "")):
                st.header("No characters found")
            else:
                st.header(license_plate_character)
        else:
            st.write("")