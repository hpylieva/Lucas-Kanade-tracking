import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import funtions

img = cv.imread('baby.jpg',0)
img2 = img.copy()
template = cv.imread('tail.jpg',0)
h, w = template.shape
# All the 6 methods for comparison in a list
methods = [
'cv.TM_SQDIFF',
'cv.TM_CCORR_NORMED'
]

for meth in methods:
    img = img2.copy()
    method = eval(meth)
    # apply template Matching
    res = cv.matchTemplate(img,template,method)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
    # if the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
   # funtions.plot_match_result(img, res, top_left, bottom_right, meth)

# separately run for SAD method as it has custom implementation
img = img2.copy()
# Apply template Matching
res = funtions.TM_SAD(img,template)
min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
top_left = max_loc
#np.unravel_index(np.argmax(res), res.shape)
print(top_left)
bottom_right = (top_left[0] + w, top_left[1] + h)
funtions.plot_match_result(img, res, top_left, bottom_right, "TM_SAD")