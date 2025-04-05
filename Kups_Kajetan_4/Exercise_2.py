import numpy as np
from pathlib import Path
import cv2

import matplotlib.pyplot as plt


#import files path
parent_dir = Path(__file__).resolve().parent
events = parent_dir / "dataset" / "events.txt"
# groundtruth_dir = parent_dir / "zaw_avs_materials" / "lab02_cfd" / "pedestrian" / "groundtruth"
# temporal_roi = parent_dir  / "zaw_avs_materials" / "lab02_cfd" / "pedestrian" / 'temporalROI.txt'

#1. read file
lines = []
with open(events, "r") as file:
    # lines = file.readlines()  # Reads all lines into a list
    # print(lines)
    i = 0 
    for line in file:
        i += 1
        words = line.strip().split(" ")  # Strip removes trailing newlines
        if i > 15000:
            break
        lines.append(words)

#2. split data
timestamp = []
x = []
y = []
p = [] #p isthe value 0 or 1 corresponding to the event polarity 

for vector in lines:
    timestamp   .append(float(vector[0]))
    x           .append(float(vector[1]))
    y           .append(float(vector[2]))
    p           .append(float(vector[3]))

print("number of events: ", len(timestamp))
print("number of events: ", len(x))
print("first timestamp: ", timestamp[0])
print("last timestamp: ", timestamp[-2])
print("max x, y: ", max(x), max(y))
number_of_negatives = p.count(0)
number_of_positives = p.count(1)
if number_of_positives > number_of_negatives:
    print("more negative events" )
else:
    print("more positive events" )


colors = ['red' if val == 1 else 'blue' for val in p]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# plt.plot(x, y, timestamp)
ax.scatter(x, y, timestamp, c=colors, marker='.', s=1)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('timestamp')

ax.set_zlim([0, 1])  

plt.show()


# 1. sequence in exercise 1 is 1 s long
# 2. it depends, but usually aboud 0.00003 s
# 3. it depends on the change of light that was capture by single pixel
# 4. positive - pixel captured brighter light, negative - darker
# 5. both directions across both axises