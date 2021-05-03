import pytesseract

def run(image):
    pytesseract.pytesseract.tesseract_cmd = '/app/.apt/usr/bin/tesseract'
    license_plate_character = pytesseract.image_to_string(image)
    return license_plate_character