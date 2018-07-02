import cv2
import numpy as np
from matplotlib import pyplot as plt


def plot_matching_result(img, match_result, top_left, bottom_right, method):
    """
    Plot results of template matching.
    Images are automatically scaled in plt.imshow() like:
    img = (img - np.min(img))/(np.max(img) - np.min(img))
    """
    cv2.rectangle(img, top_left, bottom_right, 255, 2)
    plt.subplot(211),plt.imshow(match_result,cmap = 'gray')
    plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
    plt.subplot(212),plt.imshow(img,cmap = 'gray')
    plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
    plt.suptitle("Appearance similarity function: "+str(method))
    plt.show()


def template_matching(image, template, method = 'SSD'):
    """
    Computes template matching.
    :param image: image in the grey scale
    :param templ: template in grey scale
    :param method: chosen appearance similarity function for convolution
    :return: Result of comparison the template to cutted aread from image in each point.
    The minimum pr maximum value of this result will be loaction of top left corner of image we search
    """
    im_height, im_width = image.shape
    t_height, t_width = template.shape

    res = np.zeros((im_height - t_height, im_width - t_width))
    image = np.array(image, dtype = 'float32')
    # loop through the search image
    for h in range(im_height - t_height):
        for w in range(im_width - t_width):
            patch = image[h:(h+t_height), w:(w+t_width)]
            if method=='SSD':
                res[h, w] = np.sum(np.power(patch - template,2))
            elif method=='NCC':
                res[h,w] = np.mean((patch - patch.mean())* (template - template.mean()))
                std = patch.std() * template.std()
                if std ==0:
                    res[h, w] = 0
                else:
                    res[h, w] /= std
            elif method=='SAD':
                res[h, w] = np.sum(np.absolute(patch - template))
    return res


if __name__ == "__main__":
    img = cv2.imread('TestTemplateMatching/baby.jpg', 0)
    img_copy = img.copy()
    template = cv2.imread('TestTemplateMatching/babyface.jpg', 0)
    template_copy = template.copy()
    h, w = template.shape
    # targetPoint = (162, 74)

    run_opencv_methods = False
    if run_opencv_methods:
        methods = [
            'cv2.TM_SQDIFF',
            'cv2.TM_CCORR_NORMED'
        ]

        for meth in methods:
            img = img_copy.copy()
            method = eval(meth)
            # apply template matching
            res = cv2.matchTemplate(img, template, method)
            _, _, min_loc, max_loc = cv2.minMaxLoc(res)
            # if the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
            if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
                top_left = min_loc
            else:
                top_left = max_loc
            print("Top left point of area where template match: ", top_left)
            bottom_right = (top_left[0] + w, top_left[1] + h)
            plot_matching_result(img, res, top_left, bottom_right, meth)

    methods = [
        'SSD',
        'NCC',
        'SAD'
    ]

    for method in methods:
        img = img_copy.copy()
        template = template_copy.copy()
        res = template_matching(img, template, method)
        # res = (1 - (res - np.min(res))/(np.max(res) - np.min(res)))*255
        # res = np.array(res, dtype = "uint8")
        if method in ['SSD', 'SAD']:
            top_left = np.unravel_index(res.argmin(), res.shape)[::-1]
        else:
            top_left = np.unravel_index(res.argmax(), res.shape)[::-1]
        print("Top left point of area where template match: ", top_left)
        bottom_right = (top_left[0] + w, top_left[1] + h)
        plot_matching_result(img, res, top_left, bottom_right, method)


