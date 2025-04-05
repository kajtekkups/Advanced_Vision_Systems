import cv2
from pathlib import Path
import numpy as np

from sklearn.metrics import f1_score


parent_dir = Path(__file__).resolve().parent.parent
video = parent_dir / "zaw_avs_materials" / "lab03_fos" / "pedestrians_input.mp4"


groundtruth_dir = parent_dir / "zaw_avs_materials" / "lab02_cfd" / "pedestrian" / "groundtruth"


cap = cv2.VideoCapture(video)

def calculate_indicators(groundtruth_img, detected_img, TP, TN, FP, FN):
    TP_M = np.logical_and((np.uint8(detected_img) == 255), (np.uint8(groundtruth_img) == 255)) # logical product of the matrix elements
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

def read_groundtruth(i):
    image_path = groundtruth_dir / f'gt{i:06d}.png'
    I = cv2.imread(image_path)
    I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
        
    return I


def morphologic_functions(binary_image):
    binary_image = cv2.medianBlur(binary_image, ksize=5) 
    kernel = np.ones((2,2), np.uint8)
    # eroded = cv2.erode(binary_image, kernel, iterations=1)
    dilated = cv2.dilate(binary_image, kernel, iterations=5)
    eroded = cv2.erode(dilated, kernel, iterations=2)
    
    return eroded

TP =0
TN = 0
FP = 0
FN = 0
i = 0


bgd_model = cv2.createBackgroundSubtractorKNN(history=500, dist2Threshold=600, detectShadows=False)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    subtract = bgd_model.apply(frame, learningRate=0.01) 
    subtract = morphologic_functions(subtract)

    i += 1
    groundtruth_img = read_groundtruth(i)
    groundtruth_img = np.where(groundtruth_img > 125, 255, 0).astype(np.uint8)

    # Resize both masks to the same shape
    # height, width = groundtruth_img.shape
    subtract = cv2.resize(subtract, (width, height), interpolation=cv2.INTER_NEAREST)
    subtract = subtract[0:240, :]
    groundtruth_img = groundtruth_img[:, 0:352]

    TP, TN, FP, FN = calculate_indicators(groundtruth_img, subtract, TP, TN, FP, FN)
    
    
    # Display results
    cv2.imshow("Original Frame", frame)
    cv2.imshow("groundtruth_img", groundtruth_img)
    cv2.imshow("Foreground Mask", subtract)
    cv2.waitKey(10)

P = TP / (TP + FP)
R = TP / (TP + FN)
F1 = 2 * P * R / (P + R)

print('P: ', P)
print('R: ', R)
print('F1: ', F1)
print('TP: ', TP)


cap.release()
cv2.destroyAllWindows()