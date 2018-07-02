import os
import cv2
import glob
import argparse
import numpy as np
from matplotlib import pyplot as plt


def warp_affine(image, p, inverted=False):
    affine_transform = np.float32([[1 + p[0], p[2], p[4]],
                                   [p[1], 1 + p[3], p[5]]])
    if inverted:
        affine_transform = cv2.invertAffineTransform(affine_transform)
    if len(image.shape) == 2:
        rows, cols = image.shape
        return cv2.warpAffine(image, affine_transform, (cols, rows))
    else:
        image = np.append(image, 1)
        return affine_transform.dot(image)


def cut_patch(image, region_of_interest):
    x, y, height, width = region_of_interest
    return image[y:y + height, x:x + width]


def jacobian(x):
    return [[x[0], 0, x[1], 0, 1, 0],
            [0, x[0], 0, x[1], 0, 1]]


def get_roi(target, window_size):
    return np.array([int(np.floor(target[0])) - window_size[0] // 2,
                     int(np.floor(target[1])) - window_size[1] // 2,
                     window_size[0],
                     window_size[1]])


def get_central_point(region_of_interest):
    """
    :param region_of_interest: [top_left, bottom_left, win_width, win_length]
    :return: central (target) point
    """
    window_size = region_of_interest[-2:]
    return [region_of_interest[0] + window_size[0] // 2,
            region_of_interest[1] + window_size[1] // 2]


def show_image_with_rect(image, roi):
    top_left = tuple(roi[:2])
    bottom_right = tuple([int(roi[0] + roi[2]), int(roi[1] + roi[3])])
    cv2.rectangle(image, top_left, bottom_right, 255, 2)
    cv2.imshow("Lucas-Kanade tracking",image)
    # to show sequence of images properly as a sequence
    cv2.waitKey(20)


def lucas_kanade_tracker(image_list, region_of_interest):
    num_iterations = 60

    image_start = cv2.imread(image_list.pop(0), 0)
    window_size = args.roi[-2:]
    target_point = np.float32(get_central_point(args.roi))
    target = cut_patch(image_start, region_of_interest)

    # show the first image with initial roi
    show_image_with_rect(image_start, region_of_interest)
    print(region_of_interest)

    for img in image_list:
        next_image = cv2.imread(img, 0)
        img_copy = next_image.copy()
        params = np.zeros(6, dtype="float32")
        eps = 1e-2

        for i in range(num_iterations):
            warped_img = warp_affine(next_image, params, inverted=True)
            candidate = cut_patch(warped_img, region_of_interest)

            gx = cv2.Sobel(next_image, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(next_image, cv2.CV_32F, 0, 1, ksize=3)

            gx_w = warp_affine(gx, params, inverted=True)
            gy_w = warp_affine(gy, params, inverted=True)

            gx_w = cut_patch(gx_w, region_of_interest)
            gy_w = cut_patch(gy_w, region_of_interest)

            X, Y = np.meshgrid(range(candidate.shape[0]), range(candidate.shape[1]))
            coords_2d = np.array([X.flatten(), Y.flatten()]).transpose()
            grad_image = np.array([gx_w.flatten(), gy_w.flatten()]).transpose()

            steepest_descent = []
            for i in range(grad_image.shape[0]):
                jacob = jacobian(coords_2d[i])
                steepest_descent.append(grad_image[i].dot(jacob))
            steepest_descent = np.array(steepest_descent)

            hessian = np.dot(steepest_descent.T, steepest_descent)
            error_image = np.subtract(target, candidate, dtype='float64')
            error_image_repmat = np.tile(error_image.flatten(), (len(params), 1)).T
            cost_function = np.sum(steepest_descent * error_image_repmat, axis=0)
            dp = np.dot(np.linalg.inv(hessian), cost_function.T)

            dp_norm = np.linalg.norm(dp)
            if dp_norm < eps:
                break
            else:
                params += dp.T

        # update target point, roi and target patch
        target_point = warp_affine(target_point, params)
        region_of_interest = get_roi(target_point, window_size)
        target = cut_patch(next_image, region_of_interest)

        # show image with the just found rectangle
        show_image_with_rect(img_copy, region_of_interest)
        print(region_of_interest)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--roi', nargs='+', type=int, default=[310, 102, 39, 50])
    parser.add_argument('--dpath', type=str, default='Football/img/')
    args = parser.parse_args()

    image_list = sorted(glob.glob(os.path.join(args.dpath, '*.jpg')))
    lucas_kanade_tracker(image_list, args.roi)
