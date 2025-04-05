import cv2

I = cv2.imread('mandril.jpg')

# src - Input image
# ksize - Kernel size
# sigmaX - Standard deviation in the X direction, controls bluring (if 0, OpenCV calculates it automatically)
gaussian = cv2.GaussianBlur(I, ksize=(7,7), sigmaX=0)

gaussian = cv2.GaussianBlur(I, ksize=(7,7), sigmaX=0)


# ddepth defines the data type of the output image.
sobel = cv2.Sobel(I, ddepth=cv2.CV_8U, dx=1, dy=1, ksize=5) #Calculates the first, second, third, or mixed image derivatives using an extended Sobel operator

laplacian = cv2.Laplacian(I, cv2.CV_64F, ksize=3)


median_blur = cv2.medianBlur(I, ksize=11)


cv2.imshow("Original", I)
cv2.imshow("gaussian", gaussian)
cv2.imshow("laplacian", laplacian)
cv2.imshow("sobel", sobel)
cv2.imshow("median_blur", median_blur)


cv2.waitKey()