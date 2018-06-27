import os
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


def LK(prevIm, nextIm, roi, winsize):

    pass


if __name__=="__main__":
    # img = cv.imread('baby.jpg', 0)
    # img_copy = img.copy()
    # template = cv.imread('babyface.jpg', 0)
    # template_copy = template.copy()
    datasetName = 'DragonBaby'
    imageFolder = os.getcwd() + '/' + datasetName +'/img/'
    print("Image folder ",imageFolder)
    targetPoint = [162, 72]
    window = [75, 55]
    img = cv.imread('baby.jpg', 0)
