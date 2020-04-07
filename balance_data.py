import os
from PIL import Image


number = 0

path = os.path.join("dataset", "healthy")

# For every image inside our minority folder, we're going to created a mirrored
# copy of it, save it, rotate it by 90 degrees, and save it again. We do so 4 times,
# creating 8 new files from our original one
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
