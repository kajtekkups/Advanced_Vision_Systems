from pathlib import Path
import cv2
import numpy as np


#import files path
parent_dir = Path(__file__).resolve().parent.parent
images_dir = parent_dir / "zaw_avs_materials" / "lab02_cfd" / "pedestrian" / "input"

def read_frame(i):
    image_path = images_dir / f'in{i:06d}.jpg'
    I = cv2.imread(image_path)
    I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
    I = I.astype('int')
    
    return I

def detect_object(current_frame, previous_frame):

    subtraction = ((np.int32(current_frame) - np.int32(previous_frame)) + 255)/2
    subtraction = np.uintc(subtraction)
    binary_image = np.where(subtraction > 135, 255, 0).astype(np.uint8)

    binary_image = cv2.medianBlur(binary_image, ksize=5) 
    kernel = np.ones((2,2), np.uint8)
    dilated = cv2.dilate(binary_image, kernel, iterations=7)
    eroded = cv2.erode(dilated, kernel, iterations=1)
    
    return eroded


previous_frame = read_frame(339)
for i in range(340, 1100, 1) :
    
    current_frame = read_frame(i)
    
    img = detect_object(current_frame, previous_frame)

    cv2.imshow("subtraction", img)
    cv2.waitKey(10)
    previous_frame = current_frame

