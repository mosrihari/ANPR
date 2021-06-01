import cv2
import numpy as np
from utils import local_utils

def preprocess_image(image, resize_bool=True):
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255
    if resize_bool:
        img = cv2.resize(img,(512,512))
    return img

def get_plate(img, model, Dmax=608, Dmin=256):
    vehicle = img.copy()
    ratio = float(max(vehicle.shape[:2])) / min(vehicle.shape[:2])
    side = int(ratio * Dmin)
    bound_dim = min(side, Dmax)
    _ , LpImg, _, cor = local_utils.detect_lp(model, vehicle, bound_dim, lp_threshold=0.5)
    return LpImg, cor