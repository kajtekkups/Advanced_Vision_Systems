import numpy as np
from pathlib import Path
import cv2


#import files path
parent_dir = Path(__file__).resolve().parent.parent
images_dir = parent_dir / "zaw_avs_materials" / "lab02_cfd" / "pedestrian" / "input"

class ObjectDetection:
    def __init__(self, color="red"):
            self.img = self.read_frame(200)

            self.XX, self.YY = self.img.shape #image size
            self.N =  60 # buffer size

            #buffer
            self.BUF = np . zeros ((self.XX, self.YY, self.N), np.uint8 )
            self.iN = 0 #counter BUF [: ,: , iN ] = IG ;



    def read_frame(self, i):
        image_path = images_dir / f'in{i:06d}.jpg'
        I = cv2.imread(image_path)
        I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
        
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
        kernel = np.ones((2,2), np.uint8)
        dilated = cv2.dilate(binary_image, kernel, iterations=3)
        eroded = cv2.erode(dilated, kernel, iterations=3)
        
        return eroded
    

    def alghoritm(self):
        previous_frame = self.read_frame(339)
        self.update_buffer(previous_frame)

        for i in range(340, 1100, 1) :
            current_frame = self.read_frame(i)    
            self.update_buffer(current_frame)
            
            mean = self.calculate_mean().astype(np.uint8)
            median = self.calculate_median().astype(np.uint8)

            mean_sub = cv2.subtract(mean, current_frame)
            median_sub = cv2.subtract(median, current_frame)


            mean_sub = self.morphologic_functions(mean_sub)
            median_sub = self.morphologic_functions(median_sub)


            binary_image = np.where(median_sub > 30, 255, 0).astype(np.uint8)
            binary_image_mean = np.where(mean_sub > 30, 255, 0).astype(np.uint8)
            cv2.imshow("window", binary_image)
            cv2.imshow("median", median)
            cv2.imshow("median_sub", median_sub)
            cv2.imshow("mean", binary_image_mean)
            cv2.imshow("mean_sub", mean_sub)
            cv2.waitKey(10)

            previous_frame = current_frame

# Example usage
ObjectDetectionInstance = ObjectDetection()
ObjectDetectionInstance.alghoritm()
