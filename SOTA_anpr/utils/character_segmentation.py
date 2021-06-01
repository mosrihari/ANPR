import cv2

def sort_contours(contours, reverse=False):
    i = 0
    boundingBoxes = [cv2.boundingRect(c) for c in contours]
    (cnts, boundingBoxes) = zip(*sorted(zip(contours, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))
    return cnts

def findContours(dilation_dst, plate_image):
    contours, _ = cv2.findContours(dilation_dst, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    digit_height, digit_width = 60, 30
    crop_characters = []
    image_copy = plate_image.copy()  # for visualization
    for c in sort_contours(contours):
        (x, y, w, h) = cv2.boundingRect(c)
        ratio = h / w  # height is greater than width
        if 1 <= ratio <= 3.5:  # fixed size
            if (h / plate_image.shape[0] >= 0.5):
                curr_num = dilation_dst[y:y + h, x:x + w]
                curr_num = cv2.resize(curr_num, dsize=(digit_width, digit_height))
                _, curr_num = cv2.threshold(curr_num, 220, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                crop_characters.append(curr_num)
    return crop_characters

def run(LpImg):
    if(LpImg):
        # resize
        plate_image = cv2.convertScaleAbs(LpImg[0], alpha=(255.0))
        # gray scale
        gray_img = cv2.cvtColor(plate_image,cv2.COLOR_BGR2GRAY)
        # blur image
        blurred_image = cv2.GaussianBlur(gray_img,(7,7),0)
        # binary threshold
        binary = cv2.threshold(blurred_image,180,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]
        # dialate the image to find contours
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
        dilation_dst = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel)
    crop_characters = findContours(dilation_dst, plate_image)
    return crop_characters
