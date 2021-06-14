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

def imgcrop(input, input_out):
    filename, file_extension = os.path.splitext(input)
    file_out, file_extension_out = os.path.splitext(input_out)
    
    #print("Img_crop")
    #print(filename)
    im = Image.open(input)
    imgwidth, imgheight = im.size

    #print(filename)

    xPieces = 2
    if (imgheight/imgwidth) > 1.25:
        #print("3")
        yPieces = 3
    else:
        yPieces = 2
        #print("2")

    height = imgheight // yPieces
    width = imgwidth // xPieces
    for i in range(0, yPieces):
        for j in range(0, xPieces):
            box = (j * width, i * height, (j + 1) * width, (i + 1) * height)
            #print(box)
            a = im.crop(box)
            a.save("../Datasets/Rosana2_cut_segments/no_symptoms/" + file_out + "-" + str(i) + "-" + str(j) + file_extension)


#Get datasets and init lists
data_dir = os.path.join("../Datasets/Rosana2_cut/No_symptoms")
data_dir = pathlib.Path(data_dir)

# Get total number of files
noOfFiles = 0
noOfDir = 0

for file in os.listdir(data_dir):
	file_complete = os.path.join(data_dir, file)
	imgcrop(file_complete, file)
