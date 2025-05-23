from PIL import Image
from io import BytesIO
import numpy as np
import tensorflow as tf

input_shape = (224, 224)

MODEL = None
MODEL_PATH = r"/app/model/BrainTumorPrediction.h5"

def load_model():

    global MODEL 
    if MODEL is None: 
        print(f"Attempting to load TensorFlow model from {MODEL_PATH}...")
        try:
            loaded_model_instance = tf.keras.models.load_model(MODEL_PATH)
            MODEL = loaded_model_instance 
            print("Model loaded successfully within this process.")
            return loaded_model_instance 
        except Exception as e:
            print(f"Error loading model from {MODEL_PATH}: {e}")
            raise
    else:
        print("Model already loaded in this process. Returning cached instance.")
        return MODEL 


def read_image(image_encoded: bytes):

    pil_image = Image.open(BytesIO(image_encoded))
    return pil_image

def preprocess(image: Image.Image):

    resized_image = image.resize(input_shape) 
    grayscale_image = resized_image.convert('L')
    image_array = np.array(grayscale_image)
    image_array_with_channel = np.expand_dims(image_array, axis=-1)
    processed_image_array = np.expand_dims(image_array_with_channel, axis=0)
    processed_image_array = processed_image_array / 255.0
    return processed_image_array

def predict(model_instance: tf.keras.Model, image_array: np.ndarray): 

    prediction = model_instance.predict(image_array)

    predicted_class_index = np.argmax(prediction, axis=1)[0]

    return predicted_class_index
