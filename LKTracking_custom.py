import os
import glob
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


def warp_image(input_image, p, flag='WARP_INVERSE_MAP'):
    # affine_transform = np.array([[1+p[0], p[1], 0],
    #                             [p[2], 1+ p[3], 0],
    #                             [p[4], p[5], 1]])
    affine_transform = np.array([[1 + p[0], p[2], p[4]],
                                 [p[1], 1 + p[3], p[5]]])
    # if len(input_image.shape) < 2:
    #     rows, cols = input_image.shape[0], 1
    # else:
    rows, cols = input_image.shape[:2]
    return cv.warpAffine(input_image, affine_transform, (cols, rows), flags=cv.INTER_LINEAR + cv.WARP_INVERSE_MAP)


def warp_affine_inv(image, p):
    # M = np.float32([[1, 0, x], [0, 1, y]])
    affine_transform = np.float32([[1 + p[0], p[2], p[4]],
                                 [p[1], 1 + p[3], p[5]]])
    affine_transform = cv.invertAffineTransform(affine_transform)
    if len(image.shape) == 2:
        rows, cols = image.shape
    else:
        rows, cols, _ = image.shape
    translated = cv.warpAffine(image, affine_transform, (cols, rows))
    return translated


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
    # plt.show()


def cut_patch(image, region_of_interest):
    x, y, height, width = region_of_interest
    return image[y:y + height, x:x + width]


def eval_roi(target_point, window_size):
    return [int(np.floor(target_point[0])) - window_size[0] // 2,
            int(np.floor(target_point[1])) - window_size[1] // 2,
            window_size[0],
            window_size[1]]


def jacobian(x):
    # j = lambda x: [[x[0], 0, x[1], 0, 1, 0],[ 0, x[0], 0, x[1], 0, 1]]
    # return np.apply_along_axis(j, 1, array)
    return [[x[0], 0, x[1], 0, 1, 0], [0, x[0], 0, x[1], 0, 1]]


def show_two_images(img1, img2):
    plt.subplot(211), plt.imshow(img1, cmap='gray')
    plt.subplot(212), plt.imshow(img2, cmap='gray')
    plt.show()


def LK(previous_image, next_image, target_point, window_size, num_iterations, plot_dp_mag=False, params=np.zeros(6)):
    # params = np.zeros(6)
    dp_mag = []

    region_of_interest = eval_roi(target_point, window_size)
    target = cut_patch(previous_image, region_of_interest)

    for i in range(num_iterations):
        # apply affine transform to whole image and extract patch
        warped_image = warp_affine_inv(next_image, params)
        candidate = cut_patch(warped_image, region_of_interest)

        gy, gx = np.gradient(warped_image)
        # gx = cv.Sobel(next_image, cv.CV_32F, 1, 0, ksize=3)
        # gy = cv.Sobel(next_image, cv.CV_32F, 0, 1, ksize=3)
        # show_two_images(gx, gy)

        gx_w = cut_patch(gx, region_of_interest)
        gy_w = cut_patch(gy, region_of_interest)

        X, Y = np.array(np.meshgrid(range(candidate.shape[0]), range(candidate.shape[1])), dtype='float32')
        coords2d = np.array([X.flatten('F'), Y.flatten('F')]).T
        grad_image = np.array([gx_w.flatten('F'), gy_w.flatten('F')]).T
        steepest_descent = []

        for row in range(grad_image.shape[0]):
            jacob = jacobian(coords2d[row])
            steepest_descent.append(grad_image[row].dot(jacob))

        steepest_descent = np.array(steepest_descent)
        hessian = np.dot(steepest_descent.T, steepest_descent)
        error_image = np.subtract(target, candidate, dtype='float32')
        cost_function = np.sum(np.multiply(steepest_descent, np.tile(error_image.flatten(), (len(params), 1)).T),
                               axis=0)
        dp = np.dot(np.linalg.inv(hessian), cost_function.T)
        eps = 1e-2
        dp_mag.append(np.linalg.norm(dp))
        dpmag_log.write("iter {0} dp_mag {1}\n".format(i, dp_mag[-1]))

        if dp_mag[-1] < eps:
            break
        else:
            params += dp.T

    # if plot_dp_mag:
    #     plt.plot(dp_mag)
    #     plt.show()

    return params


if __name__ == "__main__":
    dpmag_log = open("dpmag_log.txt", "w")
    datasetName = 'Coke'
    imageFolder = os.getcwd() + '/' + datasetName + '/img/'
    # print("Image folder ",imageFolder)
    imageList = sorted(glob.glob(os.path.join(imageFolder, '*.jpg')))
    image_current = np.array(cv.imread(imageList.pop(0), 0), dtype='float32')
    # cv.imread(imageList.pop(0), 0)
    # np.array(cv.imread(imageList.pop(0), 0), dtype = 'float32')

    target_point = np.array([320, 200])
    window_size = [120, 120]
    show_image_with_rect(image_current, window_size, eval_roi(target_point, window_size))

    # plt.imshow(current_image, cmap = 'gray')
    # plt.show()

    # params = np.array([0,1,0,1,0,1], dtype = 'float32')
    params = np.zeros(6)

    # iterate through images in the dataset
    for im_number, img in enumerate(imageList):
        # read new image in the flow in a grey scale
        next_image = np.array(cv.imread(img, 0), dtype='float32')
        # cv.imread(img, 0)
        # np.array(cv.imread(img, 0), dtype = 'float32')
        next_image_copy = next_image.copy()

        dpmag_log.write("\nNew image {}\n".format(im_number))
        params = LK(image_current, next_image_copy, target_point, window_size, 60, True)
        target_point_upd = update_target_point(target_point, params)

        roi_upd = eval_roi(target_point_upd, window_size)
        print(target_point_upd)
        # the next operation adds rectangle to image passed in method
        show_image_with_rect(next_image_copy, window_size, roi_upd)
        target_point = target_point_upd
        image_current = next_image

    dpmag_log.close()
