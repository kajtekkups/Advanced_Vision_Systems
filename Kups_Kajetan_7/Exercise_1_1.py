import numpy as np
import cv2
from pathlib import Path

import matplotlib.pyplot as plt
import scipy.ndimage.filters as filters



#####################################################################3
########################  import files path
#####################################################################3
parent_dir = Path(__file__).resolve().parent.parent
path = parent_dir.as_posix()
image_dir = path +  "/zaw_avs_materials/" + "lab06_fp/"
img_1_path = image_dir + "budynek1.jpg"
img_2_path = image_dir + "budynek2.jpg"


def read_frame(image_path):
    I = cv2.imread(image_path)
    I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
    return I

def filter_sobel(img, sobel_size, x, y):
    return cv2.Sobel(img, ddepth=cv2.CV_32F, dx=x, dy=y, ksize=sobel_size)


def filtert_gauss(img, gauss_filter_mask_size):
    return cv2.GaussianBlur(img, ksize=(gauss_filter_mask_size, gauss_filter_mask_size), sigmaX=0)

def find_max(image, size, threshold): # size - maximum filter mask size
    data_max = filters.maximum_filter( image, size)
    maxima = ( image == data_max )
    diff = image > threshold
    maxima[ diff == 0] = 0
    return np.nonzero(maxima)


def calculate_H(img, sobel_size, gauss_filter_mask_size, k=0.05):
    img = img.astype(np.float32)

    # calculation of I component
    partial_derivative_x = filter_sobel(img, sobel_size, x=1, y=0)
    partial_derivative_y = filter_sobel(img, sobel_size, x=0, y=1)

    partial_derivative_x = filtert_gauss(partial_derivative_x, gauss_filter_mask_size)
    partial_derivative_y = filtert_gauss(partial_derivative_y, gauss_filter_mask_size)

    I_x2 = partial_derivative_x ** 2
    I_y2 = partial_derivative_y ** 2
    I_xy = partial_derivative_x * partial_derivative_y
    
    # cv2.imshow("I_x2", I_x2)
    # cv2.imshow("I_y2", I_y2)
    # cv2.imshow("I_xy", I_xy)
    # cv2.waitKey()

    I_x2 = filtert_gauss(I_x2, gauss_filter_mask_size)
    I_y2 = filtert_gauss(I_y2, gauss_filter_mask_size)
    I_xy = filtert_gauss(I_xy, gauss_filter_mask_size)

    det_M = I_x2 * I_y2 - I_xy ** 2
    trace_M = I_x2 + I_y2
    H = det_M - k * (trace_M ** 2)

    H = cv2.normalize(H, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return H

def plot_points_on_image(image, x_points, y_points):
    img_copy = image.copy()

    # Loop through each point and plot it on the image
    for x, y in zip(x_points, y_points):
        cv2.circle(img_copy, (x, y), radius=2, color=(255, 0, 0), thickness=-1)  # Red color

    plt.imshow(img_copy, cmap='gray')
    plt.axis('off')  # Hide the axes
    plt.show()


if __name__ == "__main__":
    img_1 = read_frame(img_1_path)
    img_2 = read_frame(img_2_path)

    H_img_1 = calculate_H(img_1, sobel_size=5, gauss_filter_mask_size=5)
    H_img_2 = calculate_H(img_2, sobel_size=5, gauss_filter_mask_size=5)

    threshold_value_1 = 134  # Threshold value between 0 and 255
    _, H_img_1 = cv2.threshold(H_img_1, threshold_value_1, 255, cv2.THRESH_BINARY)
    threshold_value_2 = 121  # Threshold value between 0 and 255
    _, H_img_2 = cv2.threshold(H_img_2, threshold_value_2, 255, cv2.THRESH_BINARY)

    max_1 = find_max(H_img_1, 7, 100)
    max_2 = find_max(H_img_2, 7, 100)

    plot_points_on_image(img_1, max_1[1], max_1[0])
    plot_points_on_image(img_2, max_2[1], max_2[0])

    # cv2.imshow("img_1", img_1)
    cv2.imshow("H_img_1", H_img_1)

    # cv2.imshow("img_2", img_2)
    cv2.imshow("H_img_2", H_img_2)

    cv2.imwrite('Kups_Kajetan_7/Original.png', img_1)
    cv2.imwrite('Kups_Kajetan_7/H_img.png', H_img_1)
    cv2.waitKey()
