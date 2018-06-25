import sys
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


def plot_match_result(img, match_result, top_left, bottom_right, method_name):
    cv.rectangle(img, top_left, bottom_right, 255, 2)
    plt.subplot(121),plt.imshow(match_result,cmap = 'gray')
    plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(img,cmap = 'gray')
    plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
    plt.suptitle(method_name)
    plt.show()


def TM_SAD(	image, templ):
    """

    :param image: image in the greyscale
    :param templ:
    :return:
    """
    # minSAD = sys.float_info.max
    # bestRow, bestCol = [0,0]
    im_height, im_width = image.shape
    t_height, t_width = templ.shape

    res = np.zeros((im_height - t_height, im_width - t_width))
    # loop through the search image
    for w in range(im_height - t_height):
        for h in range(im_width - t_width):
            x,y = [0,0]
            res[w,h] = np.sum(np.absolute(image[w:(w+t_height), h:(h+t_width)] - templ))

    return res
