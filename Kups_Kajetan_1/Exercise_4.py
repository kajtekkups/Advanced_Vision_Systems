
# import matplotlib
# import matplotlib.pyplot as plt

# def rgb2gray (I):
#     return 0.299 * I[:, :, 0] + 0.587 * I[:, :, 1] + 0.114 * I[:, :, 2]

# I_HSV = matplotlib.colors.rgb_to_hsv(I)
import cv2

I = cv2.imread('mandril.jpg')

height, width = I.shape[:2] # retrieving elements 1 and 2, i.e. the corresponding height and width
scale = 2.5 # scale factor
Ix2 = cv2.resize(I ,(int(scale * height), int(scale * width)))
cv2.imshow("Big Mandrill", Ix2)

cv2.waitKey()