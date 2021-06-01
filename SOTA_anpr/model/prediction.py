import cv2
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import model_from_json
import numpy as np
# Load model architecture, weight and labels

# pre-processing input images and pedict with model
def predict_from_model(image,model,labels):
    image = cv2.resize(image,(80,80))
    image = np.stack((image,)*3, axis=-1)
    prediction = labels.inverse_transform([np.argmax(model.predict(image[np.newaxis,:]))])
    return prediction

def predict_characters(crop_characters, model, labels):
    final_string = ''
    for i, character in enumerate(crop_characters):
        title = np.array2string(predict_from_model(character, model, labels))
        final_string += title.strip("'[]")

    return final_string

def run(crop_characters):
    json_file = open('model/MobileNets_character_recognition.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights("model/License_character_recognition_weight.h5")

    labels = LabelEncoder()
    labels.classes_ = np.load('license_character_classes.npy')
    prediction = predict_characters(crop_characters, model, labels)
    return prediction
