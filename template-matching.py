import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import funtions

img = cv.imread('baby.jpg',0)
img_copy = img.copy()
template = cv.imread('babyface.jpg',0)
template_copy = template.copy()
h, w = template.shape
targetPoint = (162, 72)

run_opencv_methods = False
if run_opencv_methods:
    methods = [
        'cv.TM_SQDIFF',
        'cv.TM_CCORR_NORMED'
    ]

    for meth in methods:
        img = img_copy.copy()
        method = eval(meth)
        # apply template Matching
        res = cv.matchTemplate(img,template,method)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
        # if the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
        if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc
        print(top_left)
        bottom_right = (top_left[0] + w, top_left[1] + h)
        funtions.plot_match_result(img, res, top_left, bottom_right, meth)


methods = [
    'SSD',
    'NCC',
    'SAD'
]

for method in methods:
    img = img_copy.copy()
    template = template_copy.copy()
    res = funtions.template_matching(img, template, method)
    # res = (1 - (res - np.min(res))/(np.max(res) - np.min(res)))*255
    # res = np.array(res, dtype = "uint8")
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
    if method in ['SSD', 'SAD']:
        top_left = min_loc
    else:
        top_left = max_loc
    # print("Top left point of area where template match", top_left)
    bottom_right = (top_left[0] + w, top_left[1] + h)
    funtions.plot_match_result(img, res, top_left, bottom_right, str(method))