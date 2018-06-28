import os
import glob
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


def warp_image(input_image, p, flag ='WARP_INVERSE_MAP'):
    # affine_transform = np.array([[1+p[0], p[1], 0],
    #                             [p[2], 1+ p[3], 0],
    #                             [p[4], p[5], 1]])
    affine_transform = np.array([[1+p[0], p[2], p[4]],
                                [p[1], 1+ p[3], p[5]]])
    # if len(input_image.shape) < 2:
    #     rows, cols = input_image.shape[0], 1
    # else:
    rows, cols = input_image.shape[:2]
    return cv.warpAffine(input_image, affine_transform, (cols, rows))

def update_target_point(target_point, p):
    affine_transform = np.array([[1 + p[0], p[2], p[4]],
                                 [p[1], 1 + p[3], p[5]]])
    target_point = np.append(target_point, 1)
    return affine_transform.dot(target_point)


def show_image_with_rect(image, window_size, region_of_interest):
    top_left = tuple(region_of_interest[:2])
    bottom_right = tuple([int(top_left[i] + window_size[i]) for i in range(2)])
    cv.rectangle(image, top_left, bottom_right, 255, 2)
    plt.imshow(image, cmap='gray')
    plt.draw()
    plt.pause(0.001)


def cut_patch(image, region_of_interest):
    return image[region_of_interest[1]:region_of_interest[1] + region_of_interest[3] ,
                 region_of_interest[0]:region_of_interest[0] + region_of_interest[2] ]


def eval_roi(target_point, window_size):
    return    [int(np.floor(target_point[0])) - window_size[0] // 2,
               int(np.floor(target_point[1])) - window_size[1] // 2,
                 window_size[0],
                 window_size[1]]

def jacobian(x):
     # j = lambda x: [[x[0], 0, x[1], 0, 1, 0],[ 0, x[0], 0, x[1], 0, 1]]
     # return np.apply_along_axis(j, 1, array)
     return [[x[0], 0, x[1], 0, 1, 0],[ 0, x[0], 0, x[1], 0, 1]]


def LK(previous_image, next_image, target_point, window_size, num_iterations, plot_dp_mag = False):
    params = np.zeros(6)
    dp_mag = []

    region_of_interest = eval_roi(target_point, window_size)
    target = cut_patch(previous_image, region_of_interest)
    for i in range(num_iterations):
        # apply affine transform to whole image and extract patch
        candidate = cut_patch(warp_image(next_image, params), region_of_interest)

        gx, gy = np.gradient(next_image)
        gx_w = warp_image(gx, params)
        gy_w = warp_image(gy, params)
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
        error_image = np.subtract(target, candidate, dtype='float32')
        cost_function = np.sum(steepest_descent * np.tile(error_image.flatten(), (len(params), 1)).T, axis=0)
        dp = np.dot(np.linalg.inv(hessian), cost_function.T)
        eps = 1e-2
        dp_mag.append(np.linalg.norm(dp))
        # print("dp_mag:", dp_mag[-1])

        if dp_mag[-1] < eps:
            break
        else:
            params += dp.T

    if plot_dp_mag:
        plt.plot(dp_mag)
        plt.show()

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
    image_current = cv.imread(imageList.pop(0), 0)

    target_point = np.array([320, 200])
    window_size = [120, 120]



    # plt.imshow(current_image, cmap = 'gray')
    # plt.show()

    # params = np.array([0,1,0,1,0,1], dtype = 'float32')

    # iterate through images in the dataset
    for im_number, img in enumerate(imageList):
        # read new image in the flow in a grey scale
        next_image = cv.imread(img, 0)
        next_image_copy = next_image.copy()

        print("New image ", im_number)
        params = LK(image_current, next_image_copy, target_point, window_size, 60)
        target_point_upd = update_target_point(target_point,params)

        roi_upd = eval_roi(target_point_upd, window_size)
        print(target_point_upd)
        show_image_with_rect(next_image, window_size, roi_upd)
        target_point = target_point_upd
        image_current = next_image_copy
