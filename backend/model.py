# first neural network with keras tutorial
import numpy as np
import pandas as pd
import os
from sklearn.metrics import classification_report
import seaborn as sn; sn.set(font_scale=1.4)
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from tqdm import tqdm
##making class names 
class_names = ['']
class_names_label = {class_name:i for i, class_name in enumerate(class_names)}

nb_classes = len(class_names)

IMAGE_SIZE = (150, 150)

def load_data():
    DIRECTORY = "train"
    CATEGORY = {"train", "test"}

    output = []

    for category in CATEGORY:
        path = os.pat.join(DIRECTORY, category)
        images = []
        labels = []

        print("Loading {}".format(category))

        for folder in os.listdir(path):
            label = class_names_label[folder]

            #Iterate through each image in our folder
            for file in os.listdir(os.path.join(path, folder)):

                #Get the path name of the image
                img_path = os.path.join(os.path.join(path, folder), file)

                #Open and resize the img
                image = cv2.imread(img_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, IMAGE_SIZE)

                #Append the image and its corresponding label to the output
                images.append(image)
                label.append(label)
    images = np.array(images, dtype = 'float32')
    labels = np.array(labels, dtype = 'int32')

    output.append((images, labels))

    return output
