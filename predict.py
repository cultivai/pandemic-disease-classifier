import pathlib
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# No info messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
import tensorflow as tf
from tensorflow.keras.mixed_precision import experimental as mixed_precision

# Limit GPU memory growth for RTX cards
gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices("GPU")
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

# Enable FP16 on RTX cards for faster processing
policy = mixed_precision.Policy("mixed_float16")
mixed_precision.set_policy(policy)

from tensorflow.keras import datasets, layers, models

############################

size = 256, 256

path = "rosana"
num = 0

"""
for item in os.listdir("rosana"):
    file = os.path.join(path, item)
    im = Image.open(file)
    im = im.resize(size)
    im.save(os.path.join(path, (str(num) + ".jpg")))
    num += 1
    os.remove(file)
"""


model = models.load_model("model.h5")

for item in os.listdir("rosana"):
    img = tf.keras.preprocessing.image.load_img(os.path.join(path, item))

    img = tf.keras.preprocessing.image.img_to_array(img)

    img = np.divide(img, 255)

    img = img.reshape(-1, 256, 256, 3).astype(np.float16)
    # print(type(img))
    # plt.imshow(img)
    # plt.show()
    # print(img.shape)

    res = model.predict(img)

    print(item, res)
