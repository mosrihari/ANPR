import pytesseract

def run(image):
    license_plate_character = pytesseract.image_to_string(image)
    return license_plate_character