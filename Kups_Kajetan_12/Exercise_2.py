import matplotlib.pyplot as plt 
import cv2
import numpy as np
import Exercise_1
from sklearn import svm
import joblib

l_prob = 700
HOG_data = np.zeros([2*l_prob, 3781], np.float32)

for i in range(0, l_prob):
    IP = cv2.imread('Kups_Kajetan_12/pos/per%05d.ppm' % (i+1))
    IN = cv2.imread('Kups_Kajetan_12/neg/neg%05d.png' % (i+1))

    #train positive data
    F = Exercise_1.HOG(IP)
    HOG_data[i, 0] = 1
    HOG_data[i, 1:] = F

    #train negative data
    F = Exercise_1.HOG(IN)
    HOG_data[i+l_prob, 0] = 0
    HOG_data[i+l_prob, 1:] = F

labels = HOG_data[:, 0]
data = HOG_data[:, 1:]

svm_model = svm.SVC(kernel='linear', C=1.0)
svm_model.fit(data, labels)
model_predictions = svm_model.predict(data)


TP = 0
TN = 0
FP = 0
FN = 0

for i in range(0, len(labels)):
    if labels[i] == 1 and model_predictions[i] == 1:
        TP += 1
    elif labels[i] == 0 and model_predictions[i] == 0:
        TN += 1
    elif labels[i] == 0 and model_predictions[i] == 1:
        FP += 1
    else:
        FN += 1

accuracy = (TP + TN) / len(labels)
print('accuracy', accuracy)


# Save the classifier, predictions, and labels
joblib.dump(svm_model, 'Kups_Kajetan_12/svm_model.pkl')
joblib.dump(model_predictions, 'Kups_Kajetan_12/predictions.pkl')
joblib.dump(labels, 'Kups_Kajetan_12/labels.pkl')
