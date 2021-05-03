import cv2
import numpy as np
from PIL import Image
from util import ocr_license_plate

def run(img):
    faceCascade = cv2.CascadeClassifier('util/haarcascade_russian_plate_number.xml')
    gray = np.asarray(img.convert('L'))
    img = np.asarray(img)
    faces = faceCascade.detectMultiScale(gray,1.1,4)
    for (x,y,w,h) in faces:
        a,b = (int(0.02*img.shape[0]),int(0.025*img.shape[1]))
        plate = img[y+a:y+h-a,x+b:x+w-b,:]
        kernel = np.ones((1,1),dtype=np.uint8)
        plate = cv2.dilate(plate,kernel, iterations=1)
        plate = cv2.erode(plate, kernel, iterations=1)
        plate_gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
        thresh, plate_gray = cv2.threshold(plate_gray,127,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        license_plate_character = ocr_license_plate.run(plate_gray)
    return license_plate_character
