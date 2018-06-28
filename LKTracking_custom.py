import os
import glob
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


def affine2d(input_image, p, flag = 'WARP_INVERSE_MAP'):
    # affine_transform = np.array([[1+p[0], p[1], 0],
    #                             [p[2], 1+ p[3], 0],
    #                             [p[4], p[5], 1]])
    affine_transform = np.array([[1+p[0], p[2], p[4]],
                                [p[1], 1+ p[3], p[5]]])

    rows, cols = input_image.shape[:2]
    return cv.warpAffine(input_image, affine_transform, (cols, rows))


def show_image_with_rect(image, window_size, region_of_interest):
    top_left = tuple(region_of_interest[:2])
    bottom_right = tuple([int(top_left[i] + window_size[i]) for i in range(2)])
    cv.rectangle(image, top_left, bottom_right, 255, 2)
    plt.imshow(image, cmap='gray')
    plt.show()


def cut_patch(image, region_of_interest):
    return image[region_of_interest[1]:region_of_interest[1] + region_of_interest[3] ,
                 region_of_interest[0]:region_of_interest[0] + region_of_interest[2] ]


def jacobian(x):
     # j = lambda x: [[x[0], 0, x[1], 0, 1, 0],[ 0, x[0], 0, x[1], 0, 1]]
     # return np.apply_along_axis(j, 1, array)
     return [[x[0], 0, x[1], 0, 1, 0],[ 0, x[0], 0, x[1], 0, 1]]


def LK(prevIm, nextIm, roi, winsize):
    params = np.zeros(6)
    tform = np.linalg.inv(affine2d(params))

    return params


if __name__=="__main__":
    # img = cv.imread('baby.jpg', 0)
    # img_copy = img.copy()
    # template = cv.imread('babyface.jpg', 0)
    # template_copy = template.copy()
    datasetName = 'Coke'
    imageFolder = os.getcwd() + '/' + datasetName +'/img/'
    # print("Image folder ",imageFolder)
    imageList = sorted(glob.glob(os.path.join(imageFolder, '*.jpg')))
    image = cv.imread(imageList[0], 0)
    imageList.pop(0)

    target_point = [320, 200]
    window_size = [120, 120]

    region_of_interest = [target_point[0] - window_size[0] // 2,
                          target_point[1] - window_size[1] // 2,
                          window_size[0],
                          window_size[1]]

    target = cut_patch(image, region_of_interest)


    # plt.imshow(current_image, cmap = 'gray')
    # plt.show()
    params = np.zeros(6)
    # params = np.array([0,1,0,1,0,1], dtype = 'float32')
    num_iterations = 60
    # iterate through images in the dataset
    for img in imageList[:2]:
        # read new image in the flow in a grey scale
        next_image = cv.imread(img, 0)
        print("New image")
        for i in range(num_iterations):
            # show_image_with_rect(next_image, window_size, region_of_interest)
            # apply affine transform to whole image
            next_image = affine2d(next_image, params)
            # extract patch
            candidate = cut_patch(next_image, region_of_interest)

            gx, gy = np.gradient(next_image)
            gx_w = affine2d(gx, params)
            gy_w = affine2d(gy, params)
            gx_w = cut_patch(gx_w, region_of_interest)
            gy_w = cut_patch(gy_w, region_of_interest)
            X, Y = np.meshgrid(range(candidate.shape[0]), range(candidate.shape[1]))
            coords2d = np.array([X.flatten(), Y.flatten()]).transpose()
            grad_image = np.array([gx_w.flatten(), gy_w.flatten()]).transpose()
            steepest_descent = []

            for row in range(grad_image.shape[0]):
                jacob = jacobian(coords2d[row])
                steepest_descent.append(grad_image[row].dot(jacob))

            steepest_descent = np.array(steepest_descent)
            hessian = np.dot(steepest_descent.transpose(), steepest_descent)
            error_image = np.subtract(target, candidate, dtype = 'float32')
            cost_function = np.sum(steepest_descent * np.tile(error_image.flatten(), (len(params),1)).T, axis = 0)
            dp = np.dot(np.linalg.inv(hessian), cost_function.T)
            eps = 1e-2
            dpMag = np.sqrt(dp.T.dot(dp))
            print("dpMag:" , dpMag)

            if dpMag < eps:
                break
            else:
                params += dp.T

        # plt.imshow(image, cmap='gray')
        # plt.show()
        # gy = np.gradient(candidate,axis=0)
        show_image_with_rect(next_image, window_size, region_of_interest)


