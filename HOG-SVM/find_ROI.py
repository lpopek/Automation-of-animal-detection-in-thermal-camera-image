import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import cv2 as cv
from skimage.segmentation import felzenszwalb
from skimage.measure import regionprops
from random import randint
borderType = cv.BORDER_REPLICATE

colors = ['red', 'green']

animals_names = ['deer', 'wild_boar']


def mark_regions(img, segments, animals=None, reference_bbox=None):
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
    ax.imshow(img)
    i = 0 
    for region in regionprops(segments):
        minr, minc, maxr, maxc = region.bbox
        if check_roi_cond(region, img) is True:
            if animals[i] == 0:
                rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                            fill=False, edgecolor='red', label="deer", linewidth=2)
            if animals[i] == 1:
                rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                            fill=False, edgecolor='green',label="wild boar", linewidth=2)
            ax.add_patch(rect)
            i += 1
    for bbox in reference_bbox:
        minc, minr, maxc, maxr = bbox
        rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                            fill=False, edgecolor='blue', linewidth=2)
        ax.add_patch(rect)
    ax.legend()
    plt.show()


def norm_roi_to_square(roi):
    roi = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
    value = [randint(0, 255)]
    if roi.shape[0] > roi.shape[1]:
        param = roi.shape[0] - roi.shape[1]
        left = int(param / 2)
        right = int(param / 2)
        norm_roi = cv.copyMakeBorder(roi, 0, 0, left, right, borderType, None, value)
        #plot_img(norm_roi)
        img_norm = cv.resize(norm_roi,(128, 128))
        #plot_img(img_norm, title="obiekt znormalizowany")
        return img_norm
    elif roi.shape[0] < roi.shape[1]:
        param = roi.shape[1] - roi.shape[0]
        bottom = int(param/2)
        top = int(param/2)
        norm_roi = cv.copyMakeBorder(roi, top, bottom, 0, 0, borderType, None, value)
        #plot_img(norm_roi)
        img_norm = cv.resize(norm_roi,(128, 128))
        #plot_img(img_norm, title="obiekt znormalizowany")
        return img_norm
    else:
        #plot_img(roi)
        img_norm = cv.resize(roi,(128, 128))
        #plot_img(img_norm, title="obiekt znormalizowany")
        return roi


def check_roi_cond(region, img, small_animal=True):
    minr, minc, maxr, maxc = region.bbox
    ratio = (maxc -minc)/(maxr -minr)
    if(region.area <= 100):
        return False
    elif(abs((maxr-minr - img.shape[0])) < 10 or abs(maxc-minc - img.shape[1]) < 10):
        return False
    elif maxr - minr < 25 or maxc - minc < 25:
        return False
    elif ratio > 3 or ratio < 1/3:
        return False
    else:
        return True


def get_falzenszwalb_roi(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    seg = felzenszwalb(gray, scale=1000, sigma=4, min_size=1)
    print("Felzenszwalb's number of segments: %d" % len(np.unique(seg)))
    return seg

def get_roi(img, segments):
    roi = []
    i = 0
    seg_updated = segments
    index_to_remove = []
    for region in regionprops(segments):
        minr, minc, maxr, maxc = region.bbox
        if check_roi_cond(region, img) is True:
            cropped_image = img[minr:maxr, minc:maxc]
            #plot_img(cropped_image)
            norm_roi_to_square(cropped_image)
            roi.append(cropped_image)
        else: 
            index_to_remove.append(i)
        i += 1
    seg_updated = np.delete(seg_updated, index_to_remove)
    print(f"Reduced Number of ROI: {len(roi)}")
    return roi

def predicted_to_file(filename, animals, segments, img, confidence=1.):
    with open('./predicted_boxes/' + filename, "w") as f:
        i = 0
        for region in regionprops(segments):
            minr, minc, maxr, maxc = region.bbox
            if check_roi_cond(region, img) is True:
                f.write(f"{animals_names[animals[i]]} {confidence} {minc} {minr} {maxc} {maxr}\n")
                i += 1
        print("file saved!")
