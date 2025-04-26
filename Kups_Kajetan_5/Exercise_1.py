import numpy as np
import cv2
from pathlib import Path


#####################################################################3
########################  import files path
#####################################################################3
parent_dir = Path(__file__).resolve().parent.parent
images_dir = parent_dir / "zaw_avs_materials" / "lab04_of" 


def read_frame(i, scale_percent):
    image_path = images_dir / f'cm{i:01d}.png'
    I = cv2.imread(image_path)
    I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
    
    # Calculate the new dimensions
    width = int(I.shape[1] * scale_percent / 100)
    height = int(I.shape[0] * scale_percent / 100)
    dim = (width, height)

    return cv2.resize(I, dim, interpolation=cv2.INTER_AREA)


def block_method(I_img, J_img):
    W2 = 3 #half size of the block
    edge_case = 9
    dX = 3
    dY = 3
    size_row = I_img.shape[0]
    size_col = I_img.shape[1]

    smallest_distance_u = np.zeros((size_row, size_col))
    smallest_distance_v = np.zeros((size_row, size_col))

    for row_num in range(size_row):
        if (row_num - edge_case) < 0 or (row_num + edge_case) > size_row:
            continue
        for col_num in range(size_col):
            if (col_num - edge_case) < 0 or (col_num + edge_case) > size_col:
                continue
            
            IO = np.float32(I_img[row_num-W2:row_num+W2+1, col_num-W2:col_num+W2+1]) #block

            smallest_distance_value = np.inf

            for dY_num in range(-dY, dY):
                for dX_num in range(-dX, dX):

                    DY_ROW = row_num + dY_num
                    DX_COL = col_num + dX_num
                    JO =  np.float32(J_img[DY_ROW-W2:DY_ROW+W2+1, DX_COL-W2:DX_COL+W2+1]) #block

                    distance = np.sqrt(np.sum((np.square(JO-IO))))
                    if distance < smallest_distance_value:
                        smallest_distance_value = distance
                        smallest_distance_u[row_num][col_num] = dX_num
                        smallest_distance_v[row_num][col_num] = dY_num

    return smallest_distance_u, smallest_distance_v

def vizualization_of_optical_flow(smallest_distance_u, smallest_distance_v):
    # Convert Cartesian to Polar coordinates
    magnitude, angle = cv2.cartToPolar(smallest_distance_u, smallest_distance_v)

    # Create HSV image
    hsv = np.zeros((smallest_distance_u.shape[0], smallest_distance_u.shape[1], 3), dtype=np.uint8)

    magnitude_norm = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    
    hsv[..., 0] = angle * 90 / np.pi  # H: angle scaled to [0,180]
    hsv[..., 1] = magnitude_norm      # S: normalized magnitude
    hsv[..., 2] = 255                 # V: max brightness

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


if __name__ == "__main__":
    I = read_frame(1, 60)
    J = read_frame(2, 60)

    diff = cv2.absdiff(I, J)

    smallest_distance_u, smallest_distance_v = block_method(I, J)

    opticalFlow= vizualization_of_optical_flow(smallest_distance_u, smallest_distance_v)
    opticalFlow = cv2.resize(opticalFlow, (0,0), fx=3, fy=3, interpolation=cv2.INTER_LINEAR)

    cv2.imshow("diff", diff)
    cv2.imshow("opticalFlow", opticalFlow)
    cv2.waitKey()

