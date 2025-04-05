import numpy as np
import cv2
import matplotlib.pyplot as plt

def hist(img):
    h = np.zeros((256), np.int32) # creates and zeros single - column arrays
    for row in img:
        for x in row:
            h[x] += 1
    
    return h

I = cv2.imread('lena.png')
I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)

my_hist = hist(I)
cv2_hist = cv2.calcHist([I], [0], None, [256], [0 ,256])

plt.figure()

plt.subplot(1, 2, 1)
plt.title("My hist")
plt.bar(np.arange(256), my_hist, width=7)

plt.subplot(1, 2, 2)
#flatten used to convert from (256, 1) to 1D
#arrange uset for generating x parameter (arrey from 1 to 256)
plt.bar(np.arange(256), cv2_hist.flatten(), width=7)
plt.title("cv2 hist")
plt.show()