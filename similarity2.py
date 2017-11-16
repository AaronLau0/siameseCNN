import csv
import sys
import requests
#import skimage.io
import os
import glob
import pickle
import time
import matplotlib.pyplot as plt
from scipy.misc import imresize

from IPython.display import display, Image, HTML
from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image as kimage
import numpy as np
import pandas as pd
import scipy.sparse as sp
#import skimage.io

#sys.path.append('../')
#import helpers

def show_img(sid, img_file, img_title):
    plt.subplot(sid)
    plt.title(img_title)
    plt.xticks([])
    plt.yticks([])
    img = imresize(plt.imread(img_file), (512, 512))
    plt.imshow(img)

IMAGE_DIR = "../jpg_bw2/"#os.path.join(DATA_DIR, "jpg")
rand_img = np.random.choice(glob.glob(IMAGE_DIR + "*"))
Image(filename=rand_img)
show_img(131, rand_img, "original")
plt.show()
