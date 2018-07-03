import argparse
import glob
import os
import cv2
from copy import copy
import numpy as np


def get_rect_points(roi):
    top_left = tuple(roi[:2])
    bottom_right = (roi[0] + roi[2], roi[1] + roi[3])
    return top_left, bottom_right


def show_image_with_rect(image, roi, roi_meanshift_opencv, roi_camshift_opencv):
    top_left, bottom_right = get_rect_points(roi)
    top_left_meanshift_opencv, bottom_right_meanshift_opencv = get_rect_points(roi_meanshift_opencv)
    top_left_camshift_opencv, bottom_right_camshift_opencv = get_rect_points(roi_camshift_opencv)
    cv2.rectangle(image, top_left, bottom_right, 255, 2) #red
    cv2.rectangle(image, top_left_meanshift_opencv, bottom_right_meanshift_opencv, (0, 0, 255), 2) #blue
    cv2.rectangle(image, top_left_camshift_opencv, bottom_right_camshift_opencv, (0, 255, 0), 2) #green
    cv2.imshow("Meanshift tracking", image)
    cv2.waitKey(0)


def cut_patch(image, region_of_interest):
    x, y, height, width = region_of_interest
    return image[y:y + height, x:x + width]


def get_roi_box(target_point, window_size):
    return np.array([int(np.floor(target_point[0])) - window_size[0] // 2,
                     int(np.floor(target_point[1])) - window_size[1] // 2,
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


def mean_shift(hist, input_roi_box):
    num_of_iterations = 10
    max_distance = 0.5
    centroid = get_central_point(input_roi_box)
    input_target_point = input_roi_box[:2]
    window_size = input_roi_box[2:4]
    for i in range(num_of_iterations):
        roi_box = cut_patch(hist, get_roi_box(input_target_point + centroid, window_size))
        new_centroid = np.array([np.mean(np.argwhere(roi_box > 0)[:, 1]), np.mean(np.argwhere(roi_box > 0)[:, 0])])
        if np.linalg.norm(centroid - new_centroid) < max_distance:
            # print(i)
            break
        else:
            centroid = new_centroid
    return get_roi_box(input_target_point + centroid, window_size)


def get_roi_historgram(img, track_window):
    roi = cut_patch(img, track_window)
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
    roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
    return roi_hist


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--roi', nargs='+', type=int, default=[160,83,100,100])
    parser.add_argument('--dpath', type=str, default='DragonBaby/img/')
    args = parser.parse_args()

    image_list = sorted(glob.glob(os.path.join(args.dpath, '*.jpg')))

    window_size = args.roi[-2:]
    target_point = np.float32(get_central_point(args.roi))
    start_img = cv2.imread(image_list.pop())
    roi_box = args.roi
    roi_box_meanshift_opencv = copy(roi_box)
    roi_box_camshift_opencv = copy(roi_box)

    # show the start image
    show_image_with_rect(start_img, roi_box, roi_box_meanshift_opencv, roi_box_camshift_opencv)

    # set up the ROI histogram for tracking
    roi_hist = get_roi_historgram(start_img, roi_box)

    # specify termination criteria for tracking (obligatory for OpenCV based methods)
    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

    for img_name in image_list:
        img = cv2.imread(img_name)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
        roi_box = mean_shift(dst, roi_box)
        _, roi_box_meanshift_opencv = cv2.meanShift(dst, tuple(roi_box_meanshift_opencv), term_crit)
        _, roi_box_camshift_opencv = cv2.CamShift(dst, tuple(roi_box_camshift_opencv), term_crit)
        show_image_with_rect(img, roi_box, roi_box_meanshift_opencv, roi_box_camshift_opencv)

    cv2.destroyAllWindows()
