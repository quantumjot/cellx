import base64
import io
import os

import imageio
import numpy as np
import requests

# store the URL and token in environmental variables
# DO NOT HARDCODE THEM HERE
API_URL = os.environ["CELLX-API-URL"]
API_TOKEN = os.environ["CELLX-API-TOKEN"]


def _base64_png_to_img(x: str) -> np.ndarray:
    """Convert a base64 encoded PNG image to a numpy array.

    Parameters
    ----------
    x : str
        The base64 encoded string

    Returns
    -------
    img : np.ndarray
        The image as a numpy array

    """
    decode = base64.decodebytes(x.encode("utf-8"))
    stream = io.BytesIO(decode)
    return imageio.imread(stream, format="png")


def _img_to_base64_png(x: np.ndarray) -> str:
    """Convert a numpy array to a base64 encoded PNG image.

    Parameters
    ----------
    x : str
        The base64 encoded string

    Returns
    -------
    img : np.ndarray
        The image as a numpy array

    """
    stream = io.BytesIO()
    imageio.imsave(stream, x, format="png")
    stream_data = stream.getvalue()
    return base64.b64encode(stream_data).decode("utf-8")


def _parse_data(data: dict) -> dict:
    """Parse the data before submitting to server."""
    return data


def submit_to_server(data: dict):
    """Submit the data to the server."""
    parsed = _parse_data(data)
    response = requests.post(API_URL, data=parsed)
    return response
