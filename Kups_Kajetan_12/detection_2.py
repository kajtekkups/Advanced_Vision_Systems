import joblib
import matplotlib.pyplot as plt 
import cv2
import numpy as np
import Exercise_1
from sklearn import svm
import joblib


# Read model and labels
svm_model = joblib.load('Kups_Kajetan_12/svm_model.pkl')
labels = joblib.load('Kups_Kajetan_12/labels.pkl')

img = cv2.imread('Kups_Kajetan_12/testImage2.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def check_people_size(img_in, person_sample, resize):
    test_img = img_in.copy()

    person_size_y, person_size_x = person_sample.shape[:2]  # (height, width)
    person_size_y = int(person_size_y/resize)
    person_size_x = int(person_size_x/resize)

    x = int(test_img.shape[1] / 4)  # width coordinate
    y = int(test_img.shape[0] / 4)  # height coordinate

    # Draw rectangle (top-left corner (x, y), bottom-right corner (x + width, y + height))
    cv2.rectangle(test_img, (x, y), (x + person_size_x, y + person_size_y), (255, 20, 147), 2)

    cv2.imshow("test", test_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


resize = 0.55

person_sample = cv2.imread('Kups_Kajetan_12/pos/per00060.ppm')
person_sample = cv2.cvtColor(person_sample, cv2.COLOR_BGR2RGB)
check_people_size(img, person_sample, resize)

step = 16
img = cv2.resize(img, None, fx=resize, fy=resize)
img_copy = img.copy()

for i in range(0, int(img.shape[1]-64), step):
    for j in range(0, int(img.shape[0]-128), step):
        img_patch = img[j:128+j, i:64+i]
        feature_vector = Exercise_1.HOG(img_patch)
        model_prediction = svm_model.predict([feature_vector])

        if model_prediction == 1:            
            cv2.rectangle(img_copy, (i, j), (i+64, j+128), (255, 20, 147), 2)

# cv2.rectangle(img, (100, 100), (200, 300), (255, 20, 147), 2)
plt.figure()
plt.imshow(img_copy)
plt.savefig('output.png')
plt.show()
