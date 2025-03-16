from pathlib import Path
import cv2
import numpy as np


#import files path
parent_dir = Path(__file__).resolve().parent.parent
parent_dir = parent_dir / "zaw_avs_materials" / "lab02_cfd" / "office"
images_dir = parent_dir / "input"
groundtruth_dir = parent_dir / "groundtruth"
temporal_roi = parent_dir  / 'temporalROI.txt'

def read_frame(i):
    image_path = images_dir / f'in{i:06d}.jpg'
    I = cv2.imread(image_path)
    I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
    I = cv2.medianBlur(I, ksize=5) 
    I = I.astype('int')
    
    return I

def read_groundtruth(i):
    image_path = groundtruth_dir / f'gt{i:06d}.png'
    I = cv2.imread(image_path)
    I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
        
    return I


def detect_object(current_frame, previous_frame):

    subtraction = current_frame - previous_frame
    subtraction = abs(subtraction)
    binary_image = np.where(subtraction > 45, 255, 0).astype(np.uint8)

    kernel = np.ones((7,7), np.uint8)
    dilated = cv2.dilate(binary_image, kernel, iterations=5)
    eroded = cv2.erode(dilated, kernel, iterations=2)
    
    return eroded


def calculate_indicators(groundtruth_img, detected_img, TP, TN, FP, FN):
    TP_M = np.logical_and((detected_img == 255), (groundtruth_img == 255)) # logical product of the matrix elements
    TP_S = np.sum(TP_M) # sum of the elements in the matrix
    TP = TP + TP_S # update of the global indicator

    TN_M = np.logical_and((detected_img == 0), (groundtruth_img == 0)) # logical product of the matrix elements
    TN_S = np.sum(TN_M) # sum of the elements in the matrix
    TN = TN + TN_S # update of the global indicator

    FP_M = np.logical_and((detected_img == 255), (groundtruth_img == 0)) # a pixel belonging to the background is detected as a pixel belonging to a foreground object
    FP_S = np.sum(FP_M) # sum of the elements in the matrix
    FP = FP + FP_S # update of the global indicator

    FN_M = np.logical_and((detected_img == 0), (groundtruth_img == 255)) # a pixel belonging to an object is detected as a pixel belonging to the background.
    FN_S = np.sum(FN_M) # sum of the elements in the matrix
    FN = FN + FN_S # update of the global indicator

    return TP, TN, FP, FN


TP =0
TN = 0
FP = 0
FN = 0
previous_frame = read_frame(339)

f = open(temporal_roi, 'r') # open file
line = f.readline() # read line
roi_start, roi_end = line.split() # split line
roi_start = int(roi_start) 
roi_end = int(roi_end) 

for i in range(roi_start, roi_end, 1) :
    current_frame = read_frame(i)
    detected_img = detect_object(current_frame, previous_frame)

    groundtruth_img = read_groundtruth(i)

    TP, TN, FP, FN = calculate_indicators(groundtruth_img, detected_img, TP, TN, FP, FN)



P = TP / (TP + FP)
R = TP / (TP + FN)
F1 = 2 * P * R / (P + R)

print('P: ', P)
print('R: ', R)
print('F1: ', F1)

# result: F1:  0.7068895108940711