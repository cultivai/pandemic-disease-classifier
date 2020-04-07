import pathlib
import os
import matplotlib.pyplot as plt
import numpy as np

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

############################

data_dir = os.path.join("dataset")
data_dir = pathlib.Path(data_dir)

list_ds = tf.data.Dataset.list_files(str(data_dir / "*/*"))

image_count = len(list(data_dir.glob("*/*.jpg")))

CLASS_NAMES = np.array([item.name for item in data_dir.glob("*")])
BATCH_SIZE = 32
IMG_HEIGHT = 256
IMG_WIDTH = 256
STEPS_PER_EPOCH = np.ceil(image_count / BATCH_SIZE)


def get_label(file_path):
    # convert the path to a list of path components
    parts = tf.strings.split(file_path, os.path.sep)
    # The second to last is the class-directory
    return parts[-2] == CLASS_NAMES


def decode_img(img):
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_jpeg(img, channels=3)
    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    img = tf.image.convert_image_dtype(img, tf.float32)
    # resize the image to the desired size.
    return tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT])


def process_path(file_path):
    label = get_label(file_path)
    # load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img, label


def show_batch(image_batch, label_batch):
    plt.figure(figsize=(10, 10))
    for n in range(25):
        plt.subplot(5, 5, n + 1)
        plt.imshow(image_batch[n])
        plt.title(CLASS_NAMES[label_batch[n] == 1][0].title())
        plt.axis("off")


def prepare_for_training(ds, cache=True, shuffle_buffer_size=1000):
    # This is a small dataset, only load it once, and keep it in memory.
    # use `.cache(filename)` to cache preprocessing work for datasets that don't
    # fit in memory.
    if cache:
        if isinstance(cache, str):
            ds = ds.cache(cache)
        else:
            ds = ds.cache()

    ds = ds.shuffle(buffer_size=shuffle_buffer_size)

    # Repeat forever
    ds = ds.repeat()

    ds = ds.batch(BATCH_SIZE)

    # `prefetch` lets the dataset fetch batches in the background while the model
    # is training.
    ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return ds


def load_data():
    labeled_ds = list_ds.map(
        process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    return labeled_ds
    train_ds = prepare_for_training(labeled_ds)
    return train_ds


def show_data():
    labeled_ds = list_ds.map(
        process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    train_ds = prepare_for_training(labeled_ds)
    image_batch, label_batch = next(iter(train_ds))
    show_batch(image_batch.numpy(), label_batch.numpy())
    plt.show()
