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
img_1_path = image_dir + "left_panorama.jpg"
img_2_path = image_dir + "right_panorama.jpg"

def read_frame(image_path):
    I = cv2.imread(image_path)
    I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY).astype(np.uint8)
    return I


if __name__ == "__main__":
    left_panorama = read_frame(img_1_path)
    right_panorama = read_frame(img_2_path)

    left_orb = cv2.ORB_create()
    right_orb = cv2.ORB_create()

    # Detect keypoints and compute descriptors
    left_keypoints, left_descriptors = left_orb.detectAndCompute(left_panorama, None)
    right_keypoints, right_descriptors = left_orb.detectAndCompute(right_panorama, None)

    # Draw keypoints on the image
    left_output_image = cv2.drawKeypoints(left_panorama, left_keypoints, None, color=(0, 255, 0), flags=0)
    right_output_image = cv2.drawKeypoints(right_panorama, right_keypoints, None, color=(0, 255, 0), flags=0)

    cv2.imshow('ORB Keypoints', left_output_image)
    cv2.imshow('ORB right Keypoints', right_output_image)
    cv2.waitKey(0)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    # Match descriptors, brute force
    matches = bf.match(left_descriptors, right_descriptors)

    best_matches = sorted(matches, key=lambda x: x.distance)

    # Draw top 50 matches
    matched_img = cv2.drawMatches(left_panorama, left_keypoints, right_panorama, right_keypoints, best_matches[:50], None, flags=2)

    # Show result
    cv2.imshow('Matches', matched_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Extract keypoint coordinates
    keypointsL = np.float32([kp.pt for kp in left_keypoints])
    keypointsR = np.float32([kp.pt for kp in right_keypoints])

    # Get matched points from both images
    ptsA = np.float32([keypointsL[m.queryIdx] for m in matches])
    ptsB = np.float32([keypointsR[m.trainIdx] for m in matches])

    H, _ = cv2.findHomography(ptsA, ptsB, cv2.RANSAC)

    width = right_panorama.shape[0] + left_panorama.shape[0]
    height = right_panorama.shape[1] + left_panorama.shape[1]

    result = cv2.warpPerspective(left_panorama, H, (width, height))
    result[0: right_panorama.shape[0], 0: right_panorama.shape[1]] = right_panorama

    # Show result
    cv2.imshow('rotation', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
