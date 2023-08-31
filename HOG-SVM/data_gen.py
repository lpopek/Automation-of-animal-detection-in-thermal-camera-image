import cv2 as cv
import os
import numpy as np
import matplotlib.pyplot as plt
import random
import sklearn

def get_file_name(species_name, number, zeros_on_begin=4):
    # print(zeros_on_begin - len(str(number)))
    return f"{species_name}_" + str(number).zfill(zeros_on_begin)

def get_info(img):
    pass

def change_format_into_jpg(img):
    pass
    
def resize_img(species_name):
    path, dirs, files = next(os.walk(f"./data_set/128x128/{species_name}"))
    print(path)
    file_count = len(files)
    for i in range(1, file_count + 1):
        file_name = f"{path}/{get_file_name(species_name, i)}.jpg"
        print(f"current file: {file_name}")
        img = cv.imread(f"{file_name}")
        img_ = cv.resize(img, dsize=(128, 128))
        plt.imshow(img_)

        cv.imwrite(f"{file_name}",img_)
        # plt.show()

def flip_img_horizontaly(img):
    return cv.flip(img, 1)

def translate_img(img, x, y):
    M = np.float32([[1,0,x],[0,1,y]])
    if(len(img.shape) > 2):
        rows,cols,channel = img.shape
    else:
        raise ValueError("Obraz nie ma 3 kanałów\n")
    trans_img = cv.warpAffine(img, M,(cols,rows))
    return trans_img[x:, y:, ]

def img_gen(img, trans_rantio_max = 0.3):
    img = flip_img_horizontaly(img)
    A = None
    if random.random() > 0.5:
        minus_sign = 1
        x_trans, y_trans = random.randint(0, int(img.shape[0] * trans_rantio_max)), random.randint(0, int(img.shape[1] * trans_rantio_max))
    else:
        minus_sign = -1
        x_trans, y_trans = random.randint(0, int(img.shape[0] * trans_rantio_max)), random.randint(0, int(img.shape[1] * trans_rantio_max))
    img = translate_img(img, x_trans, y_trans)
    return img[:, :,]

def show_result(img, img_proccesed):
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(img)
    ax[1].imshow(img_proccesed)
    plt.show()
    return True

if __name__ == "__main__":
    img = cv.imread('.\photos_HOG\img_100.jpg')
    img_ = img_gen(img)
    show_result(img, img_)