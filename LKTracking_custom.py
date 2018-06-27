import os
import glob
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

def jacobian():
    pass

def affine2d(p, matrix=[]):
    affine_transform = [[1+p[0], p[1], 0],
                        [p[2], 1+ p[3], 0],
                        [p[4], p[5], 1]]
    return affine_transform


def LK(prevIm, nextIm, roi, winsize):
    params = np.zeros(6)
    tform = np.linalg.inv(affine2d(params))

    return params


if __name__=="__main__":
    # img = cv.imread('baby.jpg', 0)
    # img_copy = img.copy()
    # template = cv.imread('babyface.jpg', 0)
    # template_copy = template.copy()
    datasetName = 'DragonBaby'
    imageFolder = os.getcwd() + '/' + datasetName +'/img/'
    # print("Image folder ",imageFolder)
    imageList = sorted(glob.glob(os.path.join(imageFolder, '*.jpg')))

    targetPoint = [162, 72]
    window = [75, 55]
    current_image = cv.imread(imageList[0], 0)
    imageList.pop(0)
    # plt.imshow(current_image, cmap = 'gray')
    # plt.show()

    # iterate through images in the dataset
    for img in imageList:
        # read new image in the flow in a grey scale
        next_image = cv.imread(img, 0)
        # plt.imshow(next_image, cmap = 'gray')
        # plt.show()
        # print(img)
