import cv2 as cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.color import rgb2lab, lab2rgb


def local_pixel(img, kernel, x, y):
    m, n = kernel
    m, n = (m // 2), (n // 2)
    x_min = x - m
    x_max = x + m
    y_min = y - n
    y_max = y + n
    if y_min < 0:
        y_min = 0
    if y_max > img.shape[0]:
        y_max = img.shape[0]
    if x_min < 0:
        x_min = 0
    if x_max > img.shape[1]:
        x_max = img.shape[1]
    region = img[y_min:y_max + 1, x_min:x_max + 1]

    sum_of_region = np.sum(region)
    region_shape = region.shape
    region_element_count = region_shape[0] * region_shape[1]
    average = sum_of_region // region_element_count
    return average


def local_filter(img, kernel):
    image_shape = img.shape
    result = np.zeros(image_shape)
    for i in range(image_shape[1]):
        for j in range(image_shape[0]):
            result[i][j] = local_pixel(img, kernel, j, i)
    return result


if __name__ == '__main__':
    # Load the image and convert it to a NumPy array
    img = imread('monkey.png')
    gray = img
    # plt.imshow(np.array(img), cmap ="gray")
    # plt.show()
    print("running")
    for i in range(10):
        gray = cv2.blur(gray, (50, 50))
        #    img = local_filter(gray gray.shape)
    #    gray = img
    print("Done")
    plt.imshow(np.array(gray), cmap="gray")
    plt.show()
