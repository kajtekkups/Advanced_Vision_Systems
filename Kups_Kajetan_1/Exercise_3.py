import cv2

I = cv2.imread('mandril.jpg')

#convert to gray scale
gray_I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
HSV_I = cv2.cvtColor(I, cv2.COLOR_BGR2HSV)


H = HSV_I[:, :, 0]
S = HSV_I[:, :, 1]
V = HSV_I[:, :, 2]

cv2.imshow('window', I)
cv2.imshow('gray', gray_I)
cv2.imshow('HSV', HSV_I)

cv2.imshow('H', H)
cv2.imshow('S', S)
cv2.imshow('V', V)

cv2.waitKey()
cv2.destroyAllWindows()