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

def display_surrounding_rectangle(binary_img, original_img):
    # retval -- total number of unique labels
    # labels -- destination labeled image
    # stats -- statistics output for each label , including the background label . [x, y, width, height, area]
    # centroids -- centroid output for each label , including the background label .
    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_img)

    if ( stats.shape[0] > 1) : # are there any objects
        tab = stats[1:, 4] # 4 columns without first element
        pi = np.argmax( tab ) # finding the index of the largest item
        pi = pi + 1 # increment because we want the index in stats , not in tab (backgroun has index 0, and we omitted it two lines above)
    
    baw_original_img = np.uint8(original_img)
    # drawing a bbox
    cv2.rectangle( baw_original_img, (stats[pi, 0], stats[pi, 1]), (stats[pi, 0] + stats[pi, 2], stats[pi, 1] + stats[pi, 3]), (0 ,0 ,0), thickness=2)
    # print information about the field and the number of the largest element
    cv2.putText(baw_original_img, "%f" % stats[pi ,4], (stats[pi, 0], stats[pi, 1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
    cv2.putText(baw_original_img, "%d" % pi, (int(centroids[pi, 0]), int(centroids[pi, 1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0))

    # cv2.imshow("subtraction", np.uint8( labels / retval * 255))
    cv2.imshow("2", baw_original_img)
    cv2.waitKey()

previous_frame = read_frame(339)
for i in range(340, 1100, 1) :
    current_frame = read_frame(i)
    binary_img = detect_object(current_frame, previous_frame)

    Labled = display_surrounding_rectangle(binary_img, current_frame)


    previous_frame = current_frame

