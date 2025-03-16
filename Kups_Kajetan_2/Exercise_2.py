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
    I = cv2.medianBlur(I, ksize=5) 
    I = I.astype('int')
    
    return I

def detect_object(current_frame, previous_frame):

    subtraction = current_frame - previous_frame
    subtraction = abs(subtraction)
    binary_image = np.where(subtraction > 45, 255, 0).astype(np.uint8)

    kernel = np.ones((7,7), np.uint8)
    dilated = cv2.dilate(binary_image, kernel, iterations=5)
    eroded = cv2.erode(dilated, kernel, iterations=2)
    
    return eroded


previous_frame = read_frame(339)
for i in range(340, 1100, 1) :
    
    current_frame = read_frame(i)
    
    img = detect_object(current_frame, previous_frame)

    cv2.imshow("subtraction", img)
    cv2.waitKey()
    previous_frame = current_frame

