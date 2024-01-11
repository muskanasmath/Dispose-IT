
# Trash detection using Mask R-CNN 

import os
import sys
import random
import math
import re
import time
import glob
import skimage
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Root directory of the project. 

ROOT_DIR = os.getcwd()
print(ROOT_DIR)

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log

from trash import trash



# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Path to Trash trained weights
TRASH_WEIGHTS_PATH = "weights/mask_rcnn_trash_0200_030519_large.h5" #the best

print('Weights being used: ', TRASH_WEIGHTS_PATH)



config = trash.TrashConfig()
TRASH_DIR = 'trash'
TRASH_DIR



# Override the training configurations with a few
# changes for inferencing.

class InferenceConfig(config.__class__):
    # Run detection on one image at a time

    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()


DEVICE = "/cpu:0"  # /cpu:0 


TEST_MODE = "inference"

def get_ax(rows=1, cols=1, size=16):
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax


# Load validation dataset
dataset = trash.TrashDataset()
dataset.load_trash(TRASH_DIR, "val")

dataset.prepare()

print("Images: {}\nClasses: {}".format(len(dataset.image_ids), dataset.class_names))


# Load Model
# Create model in inference mode

with tf.device(DEVICE):
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,config=config)



# Load the weights you trained
weights_path = os.path.join(ROOT_DIR, TRASH_WEIGHTS_PATH)
model.load_weights(weights_path, by_name=True)
print("Loading weights ", TRASH_WEIGHTS_PATH)

# Get images from the directory of all the test images


jpg = glob.glob("trash_img/location1/*.jpg")
jpeg = glob.glob("trash_img/location1/*.jpeg")
jpg.extend(jpeg)
jpg


# Run detection on images


import cv2
total = 0

for image in jpg:
    print(image)
    
    image = skimage.io.imread('{}'.format(image))

    # Run object detection
    results = model.detect([image], verbose=1)
    
    
    #area estimation
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray,0,255,cv2.THRESH_OTSU + cv2.THRESH_BINARY)[1]
    pixels = cv2.countNonZero(thresh)
    pixels = (image.shape[0] * image.shape[1]) - pixels
    area = pixels/1000
    print("Area of pile = {0}".format(area))
    total += area
    #cv2.putText(image, '{}'.format(pixels), (xx,yy - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
    

    # Display results
    ax = get_ax(1)
    r = results[0]
    visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                                dataset.class_names, r['scores'], ax=ax,
                                title="Predictions")
    
avg = int(total/len(jpg))

print("Average area of the garbage pile = {0}".format(avg)) 






