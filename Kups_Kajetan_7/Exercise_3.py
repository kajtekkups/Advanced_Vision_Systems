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
img_1_path = image_dir + "fontanna1.jpg"
img_2_path = image_dir + "fontanna2.jpg"

def read_frame(image_path):
    I = cv2.imread(image_path)
    I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
    return I


def get_circle_offsets(): 
    offsets = [(0, -3), (1, -3), (2, -2), (3, -1),(3, 0), (3, 1), (2, 2), (1, 3),(0, 3), (-1, 3), (-2, 2), (-3, 1), (-3, 0), (-3, -1), (-2, -2), (-1, -3)]
    return offsets


def filtert_gauss(img, gauss_filter_mask_size):
    return cv2.GaussianBlur(img, ksize=(gauss_filter_mask_size, gauss_filter_mask_size), sigmaX=0)


def plot_points_on_image(image, points):
    img_copy = image.copy()

    # Loop through each point and plot it on the image
    for x, y, _, _ in points:
        cv2.circle(img_copy, (x, y), radius=4, color=(255, 0, 0), thickness=-1)  # Red color

    plt.imshow(img_copy, cmap='gray')
    plt.axis('off')  # Hide the axes
    plt.show()


def check_vertex(list_):
    for i in range(len(list_) - 8):  # -8 because we're checking slices of length 9
        if all(list_[i:i+9]):
            return True
    return False

def fast_detector(img, threshold):
    h, w = img.shape
    corners = []
    offsets = get_circle_offsets()
    radius = 3

    for y in range(radius, h-radius):
        for x in range(radius, w-radius):
            p = img[y, x] #central point brightness
            circle = [img[y+dy, x+dx] for dx, dy in offsets]

            # to confirm vertex, 9 consecutive pixels should be brighter or darker
            brighter = [int(val) > int(int(p) + int(threshold)) for val in circle]
            darker   = [int(val) < int(int(p) - int(threshold)) for val in circle]

            if check_vertex(brighter):
                corners.append((x, y))
            elif check_vertex(darker):
                corners.append((x, y))

    return corners


def filtert_gauss(img, gauss_filter_mask_size=5):
    return cv2.GaussianBlur(img, ksize=(gauss_filter_mask_size, gauss_filter_mask_size), sigmaX=0)


def harris_measure(img, corners, sobel_size, gauss_filter_mask_size, k=0.04):
    # image gradients
    partial_derivative_x = cv2.Sobel(img, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=sobel_size)
    partial_derivative_y = cv2.Sobel(img, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=sobel_size)

    I_x2 = partial_derivative_x ** 2
    I_y2 = partial_derivative_y ** 2
    I_xy = partial_derivative_x * partial_derivative_y
    
    I_x2 = filtert_gauss(I_x2, gauss_filter_mask_size)
    I_y2 = filtert_gauss(I_y2, gauss_filter_mask_size)
    I_xy = filtert_gauss(I_xy, gauss_filter_mask_size)

    harris_scores = []

    for x, y in corners:
        Sxx = I_x2[y, x]
        Syy = I_y2[y, x]
        Sxy = I_xy[y, x]

        # Harris
        det = Sxx * Syy - Sxy**2
        trace = Sxx + Syy
        H = det - k * (trace ** 2)
        
        harris_scores.append((x, y, H))

    return harris_scores


def non_max_suppression(points, img, window_size=3):
    h, w = img.shape
    half_circle = window_size // 2

    # Create a 2D array to store Harris score map (default -inf)
    score_map = np.full((h, w), -np.inf, dtype=np.float32)

    for x, y, score in points:
        if 0 <= x < w and 0 <= y < h:
            score_map[y, x] = score

    # Suppress non-maximum in 3x3 neighborhood
    filtered_points = []
    for x, y, score in points:
        x0 = max(0, x - half_circle)
        x1 = min(w, x + half_circle + 1)
        y0 = max(0, y - half_circle)
        y1 = min(h, y + half_circle + 1)

        local_patch = score_map[y0:y1, x0:x1]
        if score == np.max(local_patch):
            filtered_points.append((x, y, score))

    return filtered_points


def remove_border_points(points, img, patch_size=31):
    h, w = img.shape
    margin = patch_size // 2
    filtered = [(x, y, score) for x, y, score in points if margin <= x < w - margin and margin <= y < h - margin]
    return filtered


def select_top_n_points(points, N):
    sorted_points = sorted(points, key=lambda p: p[2], reverse=True)
    return sorted_points[:N]


def compute_vertex_orientation(img, keypoints, patch_size=31):
    h, w = img.shape
    r = patch_size // 2
    oriented_keypoints = []

    for x, y, score in keypoints:
        # patch borders
        x0, x1 = x - r, x + r + 1
        y0, y1 = y - r, y + r + 1

        patch = img[y0:y1, x0:x1]
        if patch.shape != (patch_size, patch_size):
            continue 

        # Create coordinate grid centered at (0, 0)
        x_coords, y_coords = np.meshgrid(np.arange(-r, r+1), np.arange(-r, r+1))

        # Compute moments
        m00 = np.sum(patch)
        if m00 == 0:
            continue  # Avoid division by zero

        m10 = np.sum(x_coords * patch)
        m01 = np.sum(y_coords * patch)

        cx = m10 / m00
        cy = m01 / m00
        theta = np.arctan2(cy, cx)

        oriented_keypoints.append((x, y, theta, score))

    return oriented_keypoints


def load_brief_pairs(path='orb_descriptor_positions.txt'):
    pairs = np.loadtxt(path, dtype=np.float32)
    return pairs.reshape(-1, 2, 2)  # shape: (256, 2, 2)


def rotate_point(x, y, angle):
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    x_rot = cos_a * x - sin_a * y
    y_rot = sin_a * x + cos_a * y
    return x_rot, y_rot


def brief_descriptor(image, oriented_keypoints, point_pairs, patch_size=31):
    r = patch_size // 2
    blurred_image = filtert_gauss(image)
    descriptors = []
    valid_keypoints = []

    for x, y, angle, score in oriented_keypoints:
        x, y = int(round(x)), int(round(y))

        # Skip if patch would go out of bounds
        if x - r < 0 or x + r >= image.shape[1] or y - r < 0 or y + r >= image.shape[0]:
            continue

        desc = []
        for (p1, p2) in point_pairs:
            # Rotate point pairs
            px1, py1 = rotate_point(p1[0], p1[1], angle)
            px2, py2 = rotate_point(p2[0], p2[1], angle)

            # Translate to image coordinates
            ix1 = int(round(x + px1))
            iy1 = int(round(y + py1))
            ix2 = int(round(x + px2))
            iy2 = int(round(y + py2))

            # Check if rotated points are still inside image
            if (0 <= ix1 < image.shape[1] and 0 <= iy1 < image.shape[0] and
                0 <= ix2 < image.shape[1] and 0 <= iy2 < image.shape[0]):
                val1 = blurred_image[iy1, ix1]
                val2 = blurred_image[iy2, ix2]
                desc.append(1 if val1 < val2 else 0)
            else:
                desc.append(0)  # default bit if out-of-bounds

        descriptors.append(np.array(desc, dtype=np.uint8))
        # valid_keypoints.append((x, y, angle, score))
        valid_keypoints.append((x, y))

    return descriptors, valid_keypoints


def match_descriptors(descs1, filtered_points_1, descs2, filtered_points_2, max_distance):
    matches = []
    similar_points = []

    for i, d1 in enumerate(descs1):
        best_dist = float('inf')
        best_j = -1
        for j, d2 in enumerate(descs2):
            dist = np.sum(d1 != d2)  # Hamming distance
            if dist < best_dist:
                best_dist = dist
                best_j = j

        if best_dist <= max_distance:
            similar_points = np.array([filtered_points_1[i], filtered_points_2[best_j]])
            matches.append((i, best_j, best_dist, similar_points))
    return matches


def select_best_matches(matches, N):
    sorted_matches = sorted(matches, key=lambda m: m[2])  # sort by distance
    return sorted_matches[:N]


def generate_brief_pairs(n=256, radius=9):
    return np.random.randint(-radius, radius + 1, (n, 2, 2)).astype(np.float32)


def plot_points_on_image(image, x_points, y_points):
    img_copy = image.copy()

    img_copy = cv2.cvtColor(img_copy, cv2.COLOR_RGB2BGR)

    # Loop through each point and plot it on the image
    for x, y in zip(x_points, y_points):
        cv2.circle(img_copy, (x, y), radius=2, color=(255, 0, 0), thickness=-1)  # Red color

    plt.imshow(img_copy)
    plt.axis('off')  # Hide the axes
    plt.show()



if __name__ == "__main__":
    img_1 = read_frame(img_1_path)
    img_2 = read_frame(img_2_path)

    treshold = 55
    points_1 = fast_detector(img_1, treshold)
    points_2 = fast_detector(img_2, treshold)
    
    # Unpack
    x_vals_1, y_vals_1 = zip(*points_1)
    x_vals_2, y_vals_2 = zip(*points_2)

    x_vals_1 = list(x_vals_1)
    y_vals_1 = list(y_vals_1)
    
    plot_points_on_image(img_1, x_vals_1, y_vals_1)
    plot_points_on_image(img_2, x_vals_2, y_vals_2)

    gauss_filter = 5
    sobel_size = 13
    best_N=200

    harris_score_points_1 = harris_measure(img_1, points_1, sobel_size=sobel_size, gauss_filter_mask_size=gauss_filter, k=0.04)
    harris_score_points_2 = harris_measure(img_2, points_2, sobel_size=sobel_size, gauss_filter_mask_size=gauss_filter, k=0.04)

    filtered_points_1 = non_max_suppression(harris_score_points_1, img_1)
    filtered_points_1 = remove_border_points(filtered_points_1, img_1)
    filtered_points_1 = select_top_n_points(filtered_points_1, N=20)
####################################################333
    #checked


    # filtered_points_1 = compute_vertex_orientation(img_1, filtered_points_1)
    descriptors_1, filtered_points_1 = brief_descriptor(img_1, filtered_points_1, generate_brief_pairs())

    filtered_points_2 = non_max_suppression(harris_score_points_2, img_2)
    filtered_points_2 = remove_border_points(filtered_points_2, img_2)
    filtered_points_2 = select_top_n_points(filtered_points_2, N=20)
    # filtered_points_2 = compute_vertex_orientation(img_2, filtered_points_2)
    descriptors_2, filtered_points_2 = brief_descriptor(img_2, filtered_points_2, generate_brief_pairs())

    matches = match_descriptors(descriptors_1, filtered_points_1, descriptors_2, filtered_points_2, max_distance=150)

    filtered_matches = select_best_matches(matches, N=10)

    simillar_points = [row[3:][0] for row in filtered_matches]
    # simillar_points = [row[[0], row[1] for row in simillar_points]
    pm.plot_matches(img_1,img_2, simillar_points)

