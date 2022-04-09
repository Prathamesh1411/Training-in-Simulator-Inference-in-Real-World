import os
import numpy as np
from PIL import Image

CLS_COLOR_MAP = np.array(
      [[0, 0, 0],
       [0, 0, 142],
       [220, 20, 60]])

def prediction_visualizer(predicted_tensor):
    predicted_pw_label = np.argmax(predicted_tensor, axis=2)
    predicted_image = CLS_COLOR_MAP[predicted_pw_label]
    image = (np.asarray(predicted_image)).astype(np.uint8)
    image = Image.fromarray(image)
    image.show()



