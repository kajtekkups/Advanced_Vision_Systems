import numpy as np
from pathlib import Path
import cv2


#import files path
parent_dir = Path(__file__).resolve().parent.parent
parent_dir = parent_dir / "zaw_avs_materials" / "lab02_cfd" / "pedestrian"
images_dir = parent_dir / "input"
groundtruth_dir = parent_dir / "groundtruth"

class ObjectDetection:
    def __init__(self, color="red"):
            self.img = self.read_frame(200)

            self.XX, self.YY = self.img.shape #image size
            self.N =  60 # buffer size

            #buffer
            self.background_model_mean = np.zeros ((self.XX, self.YY), np.uint8 )
            self.previous_background_model_mean = np.zeros ((self.XX, self.YY), np.uint8 )

            self.background_model_median = np.zeros ((self.XX, self.YY), np.float64 )
            self.previous_background_model_median = np.zeros ((self.XX, self.YY), np.float64 )
            self.a = 0.005 #weight parameter

            self.TP =0
            self.TN = 0
            self.FP = 0
            self.FN = 0
            
    
    def read_frame(self, i):
        image_path = images_dir / f'in{i:06d}.jpg'
        I = cv2.imread(image_path)
        I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY).astype(np.float64)
        
        return I


    def read_groundtruth(self, i):
        image_path = groundtruth_dir / f'gt{i:06d}.png'
        I = cv2.imread(image_path)
        I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY).astype(np.uint8)  
        return I


    #Copy pixels from source to target where mask is white (255)
    def delet_object_pixels(self, previos_img, current_img, binary_mask):
        result = np.where(binary_mask == 0, previos_img, current_img)
        
        return result
    

    def update_background_model_mean(self, current_img, binary_mask):
        self.background_model_mean = self.delet_object_pixels(self.previous_background_model_mean, self.background_model_mean, binary_mask)
        self.background_model_mean = self.a* current_img + (1- self.a)* self.previous_background_model_mean
        self.previous_background_model_mean = self.background_model_mean

    def update_background_model_median(self, current_img):
        temp_bgm = self.background_model_median
        self.background_model_median += np.where(np.uint8(self.previous_background_model_median) > np.uint8(current_img), 1, 0)
        self.background_model_median -= np.where(np.uint8(self.previous_background_model_median) < np.uint8(current_img), 1, 0)
        self.previous_background_model_median = temp_bgm
    

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


    def morphologic_functions(self, binary_image):
        binary_image = cv2.medianBlur(binary_image, ksize=5) 
        kernel = np.ones((2,2), np.uint8)
        dilated = cv2.dilate(binary_image, kernel, iterations=5)
        eroded = cv2.erode(dilated, kernel, iterations=1)
        
        return eroded
    

    def alghoritm(self):
        self.previous_background_model_mean = np.uint8(self.read_frame(139))

        # f = open(temporal_roi, 'r') # open file
        # line = f.readline() # read line
        # roi_start, roi_end = line.split() # split line
        # roi_start = int(roi_start) 

        # roi_end = int(roi_end) 
        TP =0
        TN = 0
        FP = 0
        FN = 0


        binary_image_original = np.zeros ((self.XX, self.YY), np.uint8 )
        

        for i in range(140, 1100, 1) :
            current_frame = np.uint8(self.read_frame(i))    
            
            self.update_background_model_mean(current_frame, binary_image_original)
            current_frame_median = self.read_frame(i)
            self.update_background_model_median(current_frame_median)


            mean_sub = cv2.subtract(np.uint8(self.background_model_mean), current_frame)
            median_sub = cv2.subtract(np.uint8(self.background_model_median), current_frame)


            mean_sub = self.morphologic_functions(mean_sub)
            median_sub = self.morphologic_functions(median_sub)


            groundtruth_img = self.read_groundtruth(i)

            binary_image_groundtruth_img = np.where(groundtruth_img > 40, 255, 0).astype(np.uint8)
            # median_binary_image_original = np.where(median_sub > 40, 255, 0).astype(np.uint8)
            binary_image_original = np.where(mean_sub > 40, 255, 0).astype(np.uint8)

            cv2.imshow("binary_image_mean", binary_image_original)
            cv2.imshow("binary_image_groundtruth_img", binary_image_groundtruth_img)
            cv2.imshow("median_binary_image_original", binary_image_original)
            cv2.imshow("bcg", np.uint8(self.background_model_mean))
            cv2.waitKey(10)



            TP, TN, FP, FN = self.calculate_indicators(binary_image_groundtruth_img, binary_image_original, TP, TN, FP, FN)



        P = TP / (TP + FP)
        R = TP / (TP + FN)
        F1 = 2 * P * R / (P + R)

        print('P: ', P)
        print('R: ', R)
        print('F1: ', F1)


# Example usage
ObjectDetectionInstance = ObjectDetection()
ObjectDetectionInstance.alghoritm()
