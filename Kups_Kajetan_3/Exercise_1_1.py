import numpy as np
from pathlib import Path
import cv2


#import files path
parent_dir = Path(__file__).resolve().parent.parent
images_dir = parent_dir / "zaw_avs_materials" / "lab02_cfd" / "pedestrian" / "input"
groundtruth_dir = parent_dir / "zaw_avs_materials" / "lab02_cfd" / "pedestrian" / "groundtruth"
temporal_roi = parent_dir  / "zaw_avs_materials" / "lab02_cfd" / "pedestrian" / 'temporalROI.txt'
class ObjectDetection:
    def __init__(self, color="red"):
            self.img = self.read_frame(200)

            self.XX, self.YY = self.img.shape #image size
            self.N =  60 # buffer size

            #buffer
            self.BUF = np . zeros ((self.XX, self.YY, self.N), np.uint8 )
            self.iN = 0 #counter BUF [: ,: , iN ] = IG ;

            self.TP =0
            self.TN = 0
            self.FP = 0
            self.FN = 0
            
    
    def read_frame(self, i):
        image_path = images_dir / f'in{i:06d}.jpg'
        I = cv2.imread(image_path)
        I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
        
        return I


    def read_groundtruth(self, i):
        image_path = groundtruth_dir / f'gt{i:06d}.png'
        I = cv2.imread(image_path)
        I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY).astype(np.uint8)  
        return I

    def update_buffer(self, img):
        self.BUF[:, :, self.iN] = img
        self.iN += 1
        if self.iN == self.N:
            self.iN = 0

    def calculate_mean(self):
        return np.mean(self.BUF, axis=2)
    
    def calculate_median(self):
        return np.median(self.BUF, axis=2)


    def morphologic_functions(self, binary_image):
        binary_image = cv2.medianBlur(binary_image, ksize=5) 
        kernel = np.ones((5,5), np.uint8)
        dilated = cv2.dilate(binary_image, kernel, iterations=10)
        eroded = cv2.erode(dilated, kernel, iterations=10)
        
        return eroded
    

    def calculate_indicators(self, groundtruth_img, detected_img, TP, TN, FP, FN):
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


    def alghoritm(self):
        previous_frame = self.read_frame(339)
        self.update_buffer(previous_frame)

        f = open(temporal_roi, 'r') # open file
        line = f.readline() # read line
        roi_start, roi_end = line.split() # split line
        roi_start = int(roi_start) 
        roi_end = int(roi_end) 

        TP =0
        TN = 0
        FP = 0
        FN = 0

        for i in range(340, 1100, 1) :
            current_frame = self.read_frame(i)    
            self.update_buffer(current_frame)
            
            mean = self.calculate_mean().astype(np.uint8)
            median = self.calculate_median().astype(np.uint8)

            mean_sub = cv2.subtract(mean, current_frame)
            median_sub = cv2.subtract(median, current_frame)


            mean_sub = self.morphologic_functions(mean_sub)
            median_sub = self.morphologic_functions(median_sub)


            groundtruth_img = self.read_groundtruth(i)

            binary_image_groundtruth_img = np.where(groundtruth_img > 30, 255, 0).astype(np.uint8)
            binary_image_original = np.where(median_sub > 30, 255, 0).astype(np.uint8)

            TP, TN, FP, FN = self.calculate_indicators(binary_image_groundtruth_img, binary_image_original, TP, TN, FP, FN)

            previous_frame = current_frame

        P = TP / (TP + FP)
        R = TP / (TP + FN)
        F1 = 2 * P * R / (P + R)

        print('P: ', P)
        print('R: ', R)
        print('F1: ', F1)


# Example usage
ObjectDetectionInstance = ObjectDetection()
ObjectDetectionInstance.alghoritm()
