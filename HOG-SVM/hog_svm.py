from matplotlib.pyplot import title
from skimage.feature import hog
from skimage.transform import pyramid_gaussian
from skimage.io import imread
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from skimage import color
from imutils.object_detection import non_max_suppression
import imutils
import numpy as np
import argparse
import cv2
import os
import glob
from PIL import Image # This will be used to read/modify images (can be done via OpenCV too)
from numpy import *
from plot import plot_img
import joblib
import find_ROI
import random
import xml.etree.ElementTree as ET
import sys
import keyboard

REPO_PATH = os.getcwd()
orientations = 9
pixels_per_cell = (8, 8)
cells_per_block = (2, 2)

def classify_single_img(img_test, animal_names, model):
    img_resize=resize(img_test,(128,128))
    test_img = hog(img_resize, orientations, pixels_per_cell, cells_per_block, block_norm='L2', feature_vector=True)
    l = [test_img.flatten()]
    #title = f"The predicted image is : {animal_names[model.predict(l)[0]]}"
    #plot_img(img_test, title=title)
    return model.predict(l)[0]

def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0/gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0,256)]).astype("uint8")
    return cv2.LUT(image, table)

def get_clf():
    PATH = './dataset/128x128/'
    data = []
    labels = []
    dataset = []
    animal_names_dict = {}
    animal_names_list = []
    i = 0
    for folder in os.listdir(PATH):
        print(folder)
        if folder == '.DS_Store':
            continue
        im_listing = os.listdir(PATH + str(folder))
        samples_num = size(im_listing)
        for file in im_listing:
            #img = Image.open(PATH + str(folder) + "/" + file)
            #gray = img.convert('L')
            img = cv2.imread(PATH + str(folder) + "/" + file)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            fd = hog(gray, orientations, pixels_per_cell, cells_per_block, block_norm='L2', feature_vector=True)
            data.append(fd)
            if len(fd) != 8100:
                print(fd)
                print(PATH + str(folder) + "/" + file)
            labels.append(i)
        animal_names_dict[i] = folder.replace('_', ' ')
        animal_names_list.append(folder.replace('_', ' '))
        i = i + 1
    le = LabelEncoder()
    labels = le.fit_transform(labels)
    for array in data:
        dataset.append(array)
    (train_data, test_data, train_labels, test_labels) = train_test_split(dataset, labels, test_size=0.3, random_state=50)
    model = LinearSVC()
    model.fit(train_data, train_labels)
    os.chdir(REPO_PATH)
    return model, animal_names_dict

def check_aMP(model, animals):
    os.chdir(REPO_PATH)
    os.chdir("./test-mAP/")
    cur_dir = os.getcwd()
    file_list = os.listdir()
    #random.shuffle(file_list)
    for file in file_list:
        if file.endswith('.jpg'):
            img_ = cv2.imread(file)
            #plot_img(img_, title=file)
            roi_proposals = find_ROI.get_falzenszwalb_roi(img_)
            roi_to_check = find_ROI.get_roi(img_, roi_proposals)
            roi_checked = []
            animals_checked = []
            proba_checked = []
            for item in roi_to_check:
                animal = classify_single_img(item, animals, model)
                animals_checked.append(animal)

def check_and_show_random_photo(model, animals):
    r = random.random()
    if r > 0.5:
        class_name='deer'
    else:
        class_name='wild_boar'
    os.chdir(REPO_PATH)
    os.chdir(f"./dataset_update/{class_name}/photo")
    file_list = os.listdir()
    random.shuffle(file_list)
    file = file_list[7]
    img_ = cv2.imread(file)
    plot_img(img_, title=file)
    roi_proposals = find_ROI.get_falzenszwalb_roi(img_)
    roi_to_check = find_ROI.get_roi(img_, roi_proposals)
    roi_checked = []
    animals_checked = []
    proba_checked = []
    for item in roi_to_check:
        animal = classify_single_img(item, animals, model)
        animals_checked.append(animal)
    find_ROI.mark_regions(img_, roi_proposals, animals=animals_checked)

def main():
    SVM, animals = get_clf()
    #check_aMP(SVM, animals)
    check_and_show_random_photo(SVM, animals)

if __name__ == "__main__":
    main()




