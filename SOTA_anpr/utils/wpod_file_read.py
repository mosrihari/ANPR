from tensorflow.keras.models import model_from_json

def run():
    with open("wpod-net.json", "r") as f:
        model_string = f.read()
    model = model_from_json(model_string)
    model.load_weights("wpod-net.h5")
    return model