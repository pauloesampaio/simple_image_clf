import streamlit as st
import pandas as pd
import requests
from tensorflow.image import resize, decode_jpeg
from tensorflow.keras.applications.resnet import (ResNet50,
                                                  decode_predictions,
                                                  preprocess_input)


def download_image(url, target_size=(224, 224)):
    response = requests.get(url)
    image = decode_jpeg(response.content)
    st.image(image.numpy(), width=224)
    resized_image = resize(image,
                           target_size,
                           antialias=True)
    return resized_image.numpy()


def preprocess_image(image):
    reshaped_image = image.reshape((1,) + current_image.shape)
    preprocessed_input = preprocess_input(reshaped_image)
    return preprocessed_input


def decode_result(prediction):
    decoded_prediction = decode_predictions(prediction)
    result_dict = pd.DataFrame(data=[w[2] for w in decoded_prediction[0]],
                               index=[w[1] for w in decoded_prediction[0]],
                               columns=["probability"])
    st.write(result_dict)
    return result_dict


@st.cache
def load_model():
    model = ResNet50(input_shape=(224, 224, 3))
    return model


model = load_model()

st.write('''
# Simple image clf
## Enter the image url
''')
url = st.text_input("Enter image url")
if url:
    current_image = download_image(url)
    model_input = preprocess_image(current_image)
    prediction = model.predict(model_input)
    result = decode_result(prediction)
