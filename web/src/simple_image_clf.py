import streamlit as st
import pandas as pd
import requests
from tensorflow.image import resize, decode_image
from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2,
    decode_predictions,
    preprocess_input,
)


def download_image(url, target_size=(224, 224)):
    """Function to download images from url and resize it to a target size

    Args:
        url (string): Image url.
        target_size (tuple, optional): [int]. Final size the image should be resized to. Defaults to (224, 224).

    Returns:
        np.array: Image as a numpy array
    """
    response = requests.get(url)
    image = decode_image(response.content)
    # Display image on streamlit front end
    st.image(image.numpy(), width=224)
    resized_image = resize(image, target_size, antialias=True)
    return resized_image.numpy()


def preprocess_image(image):
    """Helper function to pre process image according to the neural net documentation.

    Args:
        image (np.array): Image as a numpy array.

    Returns:
        np.array: Image as a numpy array.
    """
    reshaped_image = image.reshape((1,) + image.shape)
    preprocessed_input = preprocess_input(reshaped_image)
    return preprocessed_input


def decode_result(prediction):
    """Transform prediction indexes into human readable category

    Args:
        prediction (np.array): Batch of predictions

    Returns:
        pd.DataFrame: prediction dataframe with human readable categories.
    """
    decoded_prediction = decode_predictions(prediction)
    result_dict = pd.DataFrame(
        data=[w[2] for w in decoded_prediction[0]],
        index=[w[1] for w in decoded_prediction[0]],
        columns=["probability"],
    )
    # Display prediction result on streamlit front end
    st.write(result_dict)
    return result_dict


@st.cache
def load_model(target_size=(224, 224)):
    """Function to load neural network to memory.

    Args:
        target_size (tuple, optional): [int]. Image size to be used by the model. Defaults to (224, 224).

    Returns:
        tf.keras.Model: Keras model loaded into memory.
    """
    model = MobileNetV2(input_shape=(target_size + (3,)))
    return model


if __name__ == "__main__":
    model = load_model()

    # Write title on streamlit front end
    st.write(
        """
    # Simple image clf
    ## Enter the image url
    """
    )

    # Input box on streamlit front end
    url = st.text_input("Enter image url")
    if url:
        current_image = download_image(url)
        model_input = preprocess_image(current_image)
        prediction = model.predict(model_input)
        result = decode_result(prediction)
