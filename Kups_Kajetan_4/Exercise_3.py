import numpy as np
import cv2
from pathlib import Path
import time
#####################################################################3
########################  import files path
#####################################################################3
parent_dir = Path(__file__).resolve().parent
events = parent_dir / "dataset" / "events.txt"


#####################################################################3
########################  1. read file
#####################################################################3

def read_lines():
    lines = []
    with open(events, "r") as file:
        for line in file:
            words = line.strip().split(" ")  # Strip removes trailing newlines
            if float(words[0]) > 2:
                break
            elif float(words[0]) > 1:
                lines.append(words)

    return lines

#####################################################################
#######################  2. split data
#####################################################################

def split_data(lines):
    timestamp = []
    x = []
    y = []
    p = [] #p isthe value 0 or 1 corresponding to the event polarity 

    for vector in lines:
        timestamp   .append(float(vector[0]))
        x           .append(int(vector[1]))
        y           .append(int(vector[2]))
        p           .append(-1 if int(vector[3])==0 else 1)

    return timestamp, x, y, p


#####################################################################
#######################  2. event frame
#####################################################################
def event_frame(x, y, polarities, shape):
    img = np.full((shape[0], shape[1]), 127, dtype=np.uint8)

    for event_number in range(len(x)):
        if polarities[event_number] == -1:            
            img[x[event_number]][y[event_number]] = 0
        else:
            img[x[event_number]][y[event_number]] = 255

    return img


if __name__ == "__main__":
    
    lines = read_lines()
    timestamp, x, y, p = split_data(lines)
    tau = 0.01 #time frame period [s]

    last_timestamp = timestamp[0]
    temp_x = []
    temp_y = []
    temp_p = []
    size_col = 400
    size_row = 400
    last_frame = np.ones((size_row, size_col), dtype=np.uint8)
    all_frames = np.ones((101, size_row, size_col), dtype=np.uint8)
    i = 0
    img_number = 0
    for time_stamp in timestamp:
        temp_x.append(x[i]) 
        temp_y.append(y[i])  
        temp_p.append(p[i])  
        
        if (time_stamp - last_timestamp) > tau:
            last_frame = event_frame(temp_x, temp_y, temp_p, [size_row, size_col])
            temp_x = []
            temp_y = []
            temp_p = []
            last_timestamp = time_stamp
            all_frames[img_number] = last_frame
            img_number += 1
        i += 1

for num in range(img_number):
    cv2.imshow("all_frames", all_frames[num])
    cv2. waitKey(10)



