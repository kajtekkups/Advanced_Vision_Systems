import numpy as np
import cv2
from pathlib import Path


#####################################################################3
########################  import files path
#####################################################################3
parent_dir = Path(__file__).resolve().parent.parent
images_dir = parent_dir / "zaw_avs_materials" / "lab04_of" 


def read_frame(i, scale_percent):
    image_path = images_dir / f'cm{i:1d}.png'
    print(image_path)
    I = cv2.imread(image_path)
    I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
    
    # Calculate the new dimensions
    width = int(I.shape[1] * scale_percent / 100)
    height = int(I.shape[0] * scale_percent / 100)
    dim = (width, height)

    return cv2.resize(I, dim, interpolation=cv2.INTER_AREA)


def vizualization_of_optical_flow(smallest_distance_u, smallest_distance_v, name, img_shape=[0,0]):
    # Convert Cartesian to Polar coordinates
    magnitude, angle = cv2.cartToPolar(smallest_distance_u, smallest_distance_v)

    # Create HSV image
    hsv = np.zeros((smallest_distance_u.shape[0], smallest_distance_u.shape[1], 3), dtype=np.uint8)

    magnitude_norm = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    
    hsv[..., 0] = angle * 90 / np.pi  # H: angle scaled to [0,180]
    hsv[..., 1] = magnitude_norm      # S: normalized magnitude
    hsv[..., 2] = 255                 # V: max brightness

    RGB = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    
    cv2.imshow(name, RGB)
    cv2.waitKey()


def block_method(I_img, J_img, W2=3, dY=3, dX=3):
    edge_case = 9
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


def pyramid (im , max_scale):
    images = [im]
    for k in range(1, max_scale):
        images.append( cv2.resize(images[k -1], (0, 0), fx =0.5, fy =0.5))
    
    return images


import numpy as np

def warp_forward(img, u, v):
    H, W = u.shape
    new_img = img.copy()
    
    # Create a coordinate grid
    y, x = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')

        # x:
        # [[0 1 2 3]
        # [0 1 2 3]
        # [0 1 2 3]]

        # y:
        # [[0 0 0 0]
        # [1 1 1 1]
        # [2 2 2 2]]

    # Calculate new coordinates
    x_new = (x + u).astype(np.int32)
    y_new = (y + v).astype(np.int32)

    # Mask to keep indices within bounds
    # mask = (x_new >= 0) & (x_new < W) & (y_new >= 0) & (y_new < H)
    for row in range(H):
        for col in range(W):
            new_col = x_new[row][col]
            new_row = y_new[row][col]
            new_img[new_row][new_col] = img[row][col]

    return new_img


if __name__ == "__main__":
    I = read_frame(1, 100)
    J = read_frame(2, 100)

    diff = cv2.absdiff(I, J)

    # smallest_distance_u, smallest_distance_v = block_method(I, I, J)
    pyramid_scale = 3

    pyramid_I = pyramid(I, pyramid_scale)
    pyramid_J = pyramid(J, pyramid_scale)
    
    size_row = I.shape[0]
    size_col = I.shape[1]

    u_final = np.zeros((size_row, size_col))
    v_final = np.zeros((size_row, size_col))

    original_list = []
    wraped_list = []
    
    resize_scale = 2

    u = []
    v = []

    for i in range(1, (pyramid_scale+1)):

        smallest_distance_u, smallest_distance_v = block_method(pyramid_I[-i], pyramid_J[-i])

        u.append(smallest_distance_u)
        v.append(smallest_distance_v)

        if i < 3:
            smallest_distance_u, smallest_distance_v = block_method(pyramid_I[-i], pyramid_J[-i])
            I_new = pyramid_I[-i]
            original = pyramid_I[-i-1]

            I_new = warp_forward(pyramid_I[-i], smallest_distance_u, smallest_distance_v)
            wraped_list.append(cv2.resize(I_new, (0,0), fx=2, fy=2, interpolation=cv2.INTER_LINEAR))
            original_list.append(original)


        # cv2.imshow("original", original)
        # cv2.imshow("I_new", I_new)
        
        # cv2.waitKey()

    final_u = u[-1]
    final_v = v[-1]

    for i in range(len(u)-1, 0):
        final_u += cv2.resize(u[i], (0,0), fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
        final_v += cv2.resize(v[i], (0,0), fx=2, fy=2, interpolation=cv2.INTER_LINEAR)

    vizualization_of_optical_flow(final_u, final_v, "final optical flow")
    
    cv2.imshow("original_list_1", original_list[0])
    cv2.imshow("wraped_list_1", wraped_list[0])
    cv2.imshow("original_list_2", original_list[1])
    cv2.imshow("wraped_list_2", wraped_list[1])
    cv2.waitKey()
    # opticalFlow= vizualization_of_optical_flow(smallest_distance_u, smallest_distance_v)


    # cv2.imshow("diff", diff)
    # cv2.imshow("opticalFlow", opticalFlow)
    # cv2.waitKey()
