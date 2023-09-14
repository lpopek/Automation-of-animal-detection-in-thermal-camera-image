from skimage.feature import hog
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
import numpy as np
import cv2
import os,sys
from numpy import *

from generate_ground_truth import convert_xml2yolo
import find_ROI


REPO_PATH = os.getcwd()
orientations = 9
pixels_per_cell = (8, 8)
cells_per_block = (2, 2)

def classify_single_img(img_test, animal_names, model):
    img_resize=resize(img_test,(128,128))
    test_img = hog(img_resize, orientations, pixels_per_cell, cells_per_block, block_norm='L2', feature_vector=True)
    l = [test_img.flatten()]
    return model.predict(l)[0]

def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0/gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0,256)]).astype("uint8")
    return cv2.LUT(image, table)

def get_clf():
    PATH = './HOG-SVM/128x128'
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
        im_listing = os.listdir(PATH + '/' +str(folder))
        for file in im_listing:
            img = cv2.imread(PATH + '/' + str(folder) + '/' + file)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            fd = hog(gray, orientations, pixels_per_cell, cells_per_block, block_norm='L2', feature_vector=True)
            data.append(fd)
            if len(fd) != 8100:
                print(fd)
                print(PATH + str(folder) + '/' + file)
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

def check_photo(model, animals, file_name):
    if not os.path.exists(f'HOG-SVM/examples/{file_name}.jpg'):
        raise FileNotFoundError('[Error]: File with image not found.')
    img_ = cv2.imread(f'HOG-SVM/examples/{file_name}.jpg')
    if not os.path.exists(f'HOG-SVM/examples/{file_name}.xml'):
        raise FileNotFoundError('[Error]: File with reference bbox not found.')
    bbox_list = convert_xml2yolo(f'HOG-SVM/examples/{file_name}.xml')
    roi_proposals = find_ROI.get_falzenszwalb_roi(img_)
    roi_to_check = find_ROI.get_roi(img_, roi_proposals)
    animals_checked = []
    for item in roi_to_check:
        animal = classify_single_img(item, animals, model)
        animals_checked.append(animal)
    find_ROI.mark_regions(img_, roi_proposals, animals=animals_checked, reference_bbox=bbox_list)

    
def main(file_name):
    SVM, animals = get_clf()
    check_photo(SVM, animals, file_name)

if __name__ == "__main__":
    try:
        if len(sys.argv) > 1:
            main(sys.argv[1])
        else:
            raise FileNotFoundError('[Error]: Write file name.')
    except FileNotFoundError as e:
        print(f'{e}')
    