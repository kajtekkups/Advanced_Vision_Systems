import cv2
import numpy as np
import matplotlib.pyplot as plt

I = cv2.imread('lena.png')

I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)

hist, bins = np.histogram(I.flatten(), 256, [0, 256])

classic_equalised = cv2.equalizeHist(I)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
clashe_equalized = clahe.apply(I)
# clipLimit - maximum height of the histogram bar - values above are distributed among neighbours
# tileGridSize - size of a single image block ( local method , operates on separate image blocks )

cv2.imshow('classic_equalised', classic_equalised)
cv2.imshow('clashe_equalized', clashe_equalized)
cv2.imshow('original', I)

cv2.waitKey()

I_hist = cv2.calcHist(I, [0], None, [256], [0 ,256])
classic_equalised_hist = cv2.calcHist(classic_equalised, [0], None, [256], [0 ,256])
clashe_equalized_hist = cv2.calcHist(clashe_equalized, [0], None, [256], [0 ,256])

plt.figure()
plt.subplot(1, 3, 1)
plt.title("original")
plt.bar(np.arange(256), I_hist.flatten(), width=7)
plt.subplot(1, 3, 2)
plt.title("classic")
plt.bar(np.arange(256), classic_equalised_hist.flatten(), width=7)
plt.subplot(1, 3, 3)
plt.title("clahe")
plt.bar(np.arange(256), clashe_equalized_hist.flatten(), width=7)
plt.show()

