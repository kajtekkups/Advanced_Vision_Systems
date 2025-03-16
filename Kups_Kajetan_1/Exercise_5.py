import cv2
import numpy as np

mandril = cv2.imread('mandril.jpg')
lena = cv2.imread('lena.png')

#convert to gray scale
mandril_gray = cv2.cvtColor(mandril, cv2.COLOR_BGR2GRAY)
lena_gray = cv2.cvtColor(lena, cv2.COLOR_BGR2GRAY)

addition = mandril_gray + lena_gray
subtruction = mandril_gray - lena_gray
multiplication = mandril_gray * lena_gray
module = cv2.absdiff(mandril_gray, lena_gray)

cv2.imshow('addition', addition)
cv2.imshow('subtruction', subtruction)
cv2.imshow('multiplication', multiplication)
cv2.imshow('module', module)

#LINEAR COMBINATIOM
linear_combination = 0.5*lena + 1.4*mandril 
cv2.imshow('linear_combination', np.uint8(linear_combination))


cv2.waitKey()
cv2.destroyAllWindows()