import pathlib
import os
import matplotlib.pyplot as plt
import numpy as np
from load_dataset import load_data

# No info messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
import tensorflow as tf

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


from tensorflow.keras.mixed_precision import experimental as mixed_precision
from tensorflow.keras import datasets, layers, models

#########################################

data_dir = os.path.join("dataset")
data_dir = pathlib.Path(data_dir)

list_ds = tf.data.Dataset.list_files(str(data_dir / "*/*"))

image_count = len(list(data_dir.glob("*/*.jpg")))

CLASS_NAMES = np.array([item.name for item in data_dir.glob("*")])
BATCH_SIZE = 32
IMG_HEIGHT = 256
IMG_WIDTH = 256
STEPS_PER_EPOCH = np.ceil(image_count / BATCH_SIZE)

ds = load_data()

