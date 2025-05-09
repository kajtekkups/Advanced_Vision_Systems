import numpy as np
import cv2
from pathlib import Path

import matplotlib.pyplot as plt
import scipy.ndimage.filters as filters
from scipy.ndimage import maximum_filter

# import sys
# sys.path.append('../zaw_avs_materials/lab06_fp')
import pm


#####################################################################3
########################  import files path
#####################################################################3
parent_dir = Path(__file__).resolve().parent.parent
path = parent_dir.as_posix()
image_dir = path +  "/zaw_avs_materials/" + "lab06_fp/"
img_1_path = image_dir + "eiffel1.jpg"
img_2_path = image_dir + "eiffel2.jpg"

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


def extract_neighbourhoods(img, x_points, y_points, neighbourhood_size, X_img, Y_img):
    guard = neighbourhood_size
    points = list (filter(lambda pt: pt[1] >= guard and pt[1] < Y_img - guard and pt[0] >= guard and pt[0] < X_img - guard, zip(x_points, y_points)))

    neighbourhoods = []

    for x, y in points:
        # Extract the patch around the point
        top_left = (x - neighbourhood_size, y - neighbourhood_size)
        bottom_right = (x + neighbourhood_size, y + neighbourhood_size)

        # Extract the patch around the point
        patch = img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

        flattened_patch = patch.flatten()

        # Add the flattened patch and the central coordinates to the result
        neighbourhoods.append((flattened_patch, (y, x)))

    return neighbourhoods

def patches_simmilarity_meassure(patch1, patch2):
    # Mean Squared Error (MSE) to measure the difference between two patches.
    return np.mean((patch1 - patch2) ** 2)


def compare_neighbourhoods(neighbourhoods1, neighbourhoods2, n):
    similarities = []

    # Compare each patch from neighbourhoods1 with each patch from neighbourhoods2
    for i, (patch1, _) in enumerate(neighbourhoods1):
        for j, (patch2, _) in enumerate(neighbourhoods2):            
            similarity_score = patches_simmilarity_meassure(patch1, patch2) # Compute similarity  between the two patches
            # similarities.append((similarity_score, neighbourhoods1[i], neighbourhoods2[j]))
            similarities.append((similarity_score, neighbourhoods1[i][1], neighbourhoods2[j][1]))
    
    # Sort the similarities by the score
    similarities.sort(key=lambda x: x[0])

    # Return the top n most similar pairs
    return similarities[:n]


if __name__ == "__main__":
    img_1 = read_frame(img_1_path)
    img_2 = read_frame(img_2_path)

    gauss_filter = 5
    sobel_filter = 5
    H_img_1 = calculate_H(img_1, sobel_size=sobel_filter, gauss_filter_mask_size=gauss_filter)
    H_img_2 = calculate_H(img_2, sobel_size=sobel_filter, gauss_filter_mask_size=gauss_filter)

    threshold_value_1 = 140  # Threshold value between 0 and 255
    _, H_img_1 = cv2.threshold(H_img_1, threshold_value_1, 255, cv2.THRESH_BINARY)
    threshold_value_2 = 130  # Threshold value between 0 and 255
    _, H_img_2 = cv2.threshold(H_img_2, threshold_value_2, 255, cv2.THRESH_BINARY)
    
    max_1 = find_max(H_img_1, 7, 150)
    max_2 = find_max(H_img_2, 7, 140)

    Y_img , X_img = img_1.shape
    neib_size = 7
    neighbourhood_1 = extract_neighbourhoods(img_1, max_1[1], max_1[0], neib_size, X_img, Y_img)
    neighbourhood_2 = extract_neighbourhoods(img_2, max_2[1], max_2[0], neib_size, X_img, Y_img)

    simillar_points = compare_neighbourhoods(neighbourhood_1, neighbourhood_2, 150)
    simillar_points = [row[1:] for row in simillar_points]
    
    # simillar_points = [((np.int64(18), np.int64(683)), (np.int64(17), np.int64(238))), ((np.int64(18), np.int64(683)), (np.int64(17), np.int64(537))), ((np.int64(18), np.int64(683)), (np.int64(18), np.int64(263))), ((np.int64(18), np.int64(683)), (np.int64(18), np.int64(652))), ((np.int64(18), np.int64(683)), (np.int64(19), np.int64(624))), ((np.int64(18), np.int64(683)), (np.int64(19), np.int64(687))), ((np.int64(18), np.int64(683)), (np.int64(20), np.int64(616))), ((np.int64(18), np.int64(683)), (np.int64(21), np.int64(489))), ((np.int64(18), np.int64(683)), (np.int64(22), np.int64(586))), ((np.int64(18), np.int64(683)), (np.int64(23), np.int64(611)))]
    # print(simillar_points)
    pm.plot_matches(img_1,img_2, simillar_points)

