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
class_names = ['Acer palmatum', 'aesculus chinesis', 'Albizia julibrissin', 'Aucuba japonica var. variegata', 'Buxus sinica var. parvifolia', 
'Camptotheca acuminata', 'Cedrus deodara', 'Celtis sinensis', 'cinnamomum camphora (Linn) Presl', 'Elaeocarpus decipiens'
'Euonymus japonicus', 'Euoynmus japonicus Aureo_marginatus', 'Flowering cherry', 'Ginkgo biloba', 'Juniperus cinensis Kaizuca'
'Koelreuteria paniculata', 'Lagerstroemia indica', 'Ligustrum lucidum', 'Liquidambar formosana', 'Liriodendron chinense',
'Llex cornuta', 'Loropetalum chinense var. rubrum', 'Magnolia grandiflora L', 'Magnolia liliflora Desr', 'Malushaliana',
'Metasequoia glyptostroboides', 'Michelia chapensis', 'Michelia figo (Lour.) Spreng', 'Nandina domestica', 'Nerium oleander L',
'Osmanthus fragrans', 'Photinia serratifolia', 'Pinus massoniana Lamb', 'Pinus parviflora', 'Pittosporum tobira',
'Platanus', 'Platycladus orietalis Beverlevensis', 'Pordocarpus macrophyllus', 'Populus L','Prunus cerasifera f. atropurpurea',
'Prunus persica', 'Rhododendron pulchrum', 'Sabina chinensis cv. Pyramidalis', 'Salix babylonica', 'Sapindus saponaria',
'Styphnolobium japonicum', 'Taxodium ascendens Brongn', 'Triadica sebifera', 'Viburnum odoratissimum', 'Zelkova serrata']
class_names_label = {class_name:i for i, class_name in enumerate(class_names)}

nb_classes = len(class_names)

IMAGE_SIZE = (150, 150)

def load_data():
    # Relative path to the data directory
    DIRECTORY = "../train"

    # Enumerate through each folder, with the folder
    # name being one of the catagories
    CATEGORY = []

    # Iterate through each folder in the directory
    for folder in os.listdir(DIRECTORY):
        CATEGORY.append(str(folder))

    output = []

    for category in CATEGORY:
        path = os.path.join(DIRECTORY, category)
        images = []
        labels = []

        print("Loading {}".format(category))

        #Iterate through each image in our folder
        for file in os.listdir(path):

            #Get the path name of the image
            img_path = os.path.join(path, file)

            #Open and resize the img
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, IMAGE_SIZE)

            #Append the image and its corresponding label to the output
            images.append(image)
            labels.append(category)
            
    images = np.array(images, dtype = 'float32')
    labels = np.array(labels, dtype = 'int32')

    output.append((images, labels))

    return output

if __name__ == "__main__":
    load_data()