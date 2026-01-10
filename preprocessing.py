# preprocessing.py
import numpy as np
import cv2

def load_data(image_path, npy_path):
    data = np.load(npy_path)
    if data.ndim > 2:
        data = data.squeeze()
    if data.ndim != 2:
        raise ValueError(f"Expected 2D sonar matrix, got shape {data.shape}")

    if image_path is not None:
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"PNG could not be read: {image_path}")
        return image, data

    # fallback: generate image from sonar
    m = data.astype(np.float32)
    m -= m.min()
    if m.max() > 0:
        m /= m.max()
    m = (m * 255).astype(np.uint8)
    image = cv2.applyColorMap(m, cv2.COLORMAP_JET)
    return image, data

def preprocess(image, data):
    return image, data
