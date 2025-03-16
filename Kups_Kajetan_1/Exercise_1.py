import cv2

I = cv2.imread('mandril.jpg')
# df = pd.DataFrame
print(I.shape)
print(I.size)
print(I.dtype)

cv2.imshow('window', I)
cv2.waitKey()
cv2.destroyAllWindows()

cv2.imwrite("test1.png", I)

print("hello")