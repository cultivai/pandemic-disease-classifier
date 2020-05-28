
#!/usr/bin/env python
import pathlib
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import time
import pandas as pd
from sklearn.metrics import average_precision_score

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

# Resize images to fit into our input layer if needed
size = 256, 256

# Test images path
path = "test"
num = 0

# Resize each test image
for item in os.listdir(path):
    file = os.path.join(path, item)
    if("Healthy" in file):
        label = "Healthy"
    else:
        label = "Bad"
#    print(file)
    im = Image.open(file)
    im = im.resize(size)
    im.save(os.path.join(path, (str(num) + label + ".jpg")))
    num += 1
 #   print(file)
    os.remove(file)

print("Terminei resize")

# Loads our model and does a prediction for everyfile inside our folder
model = models.load_model("model_resnet_PlantVillage.h5")

# Init variavles and time
imgs = []
results = []
start = time.time()
array_pred = []
array_actu = []

# For each inference image
for item in os.listdir("test"):

    # Load and reshape each image
    name = os.path.join(path,item)
    img = tf.keras.preprocessing.image.load_img(name)
    if ("Healthy" in name):
        array_actu.append(1)
    else:
        array_actu.append(0)
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = np.divide(img, 255)
    imgs.append(img)
    img = img.reshape(-1, 256, 256, 3).astype(np.float16)
    # print(type(img))
    # plt.imshow(img)
    # plt.show()
    # print(img.shape)

    # Predict result from image using model
    res = model.predict(img)
    results.append(res[0])

print(array_actu)

# Total inference time
print("Took ", time.time() - start, " seconds")

# Plot results in an image
plt.figure()
for i in range(len(imgs)):
    plt.imshow(imgs[i], cmap=plt.cm.binary)
    label = "Good" if results[i] > 0.7 else ("Bad" if results[i] < 0.5 else "Not sure")
    if (label == "Good"):
        array_pred.append(1)
    else:
        array_pred.append(0)

    label += " (" + str(results[i][0]) + ")"
#    plt.axis('off')
    plt.xlabel(label)
    plt.yticks([])
    plt.xticks([])
    plt.savefig("results/"+label+"_"+str(i)+".png", bbox_inches="tight")

print(array_pred)

y_actu = pd.Series(array_actu, name='Actual')
y_pred = pd.Series(array_pred, name='Predicted')

df_confusion = pd.crosstab(y_actu, y_pred, rownames=['Actual'], colnames=['Predicted'], margins=True)
print(df_confusion)

y_true = np.array(array_actu)
y_score = np.array(array_pred)
print(average_precision_score(y_true, y_score))
