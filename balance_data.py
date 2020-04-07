import os
from PIL import Image


labels = ["healthy", "bad"]

number = 0

path = os.path.join("dataset", "healthy")

for file in os.listdir(path):
    filename = os.path.join(path, file)
    im = Image.open(filename)

    for _ in range(4):
        trans = im.transpose(Image.FLIP_LEFT_RIGHT)

        trans.save(os.path.join(path, (str(number) + ".jpg")))
        number += 1
        im = im.rotate(90)
        im.save(os.path.join(path, (str(number) + ".jpg")))
        number += 1

    os.remove(filename)
