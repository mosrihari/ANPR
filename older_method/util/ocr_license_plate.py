import pytesseract

def run(image):
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    license_plate_character = pytesseract.image_to_string(image)
    return license_plate_character