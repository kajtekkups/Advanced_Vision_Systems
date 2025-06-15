import matplotlib.pyplot as plt
import cv2
import numpy as np
import scipy.ndimage
import math
from PIL import Image


def gradient(im):
    # compute the Gradient Components
    dx = scipy.ndimage.convolve1d(np.int32(im), np.array([-1, 0, 1]), 1)
    dy = scipy.ndimage.convolve1d(np.int32(im), np.array([-1, 0, 1]), 0)

    # gradient amplitude
    grad = np.sqrt(dx ** 2 + dy ** 2)
    grad = grad / np.amax(grad)

    orient = np.arctan2(dy, dx)
    return grad, orient

def grad_rgb(im):
    # each RGB channel should be counted separately
    magnitude = np.zeros_like(im, dtype=float)
    orientation = np.zeros_like(im, dtype=float)

    magnitude[:, :, 2] , orientation[:, :, 2] = gradient(im[:, :, 2])
    magnitude[:, :, 1] , orientation[:, :, 1] = gradient(im[:, :, 1])
    magnitude[:, :, 0] , orientation[:, :, 0] = gradient(im[:, :, 0])

    max_magnitude = np.max(magnitude, axis=2)
    # channel  that was maximal
    max_index = np.argmax(magnitude, axis=2)

    
    # we take the orientation for which gradient magnitude was the highest
    max_orientation = np.take_along_axis(orientation, np.expand_dims(max_index, axis=2), axis=2)
    max_orientation = np.squeeze(max_orientation, axis=2)  # remove redundant dimension (make it 2d)

    return max_magnitude, max_orientation


def create_HOG_img(histogram, cell_size):
    initial_pattern = np.zeros((cell_size, cell_size))
    initial_pattern[np.round(cell_size // 2).astype(int):np.round(cell_size // 2).astype(int) + 1, :] = 1  # create a center line
    visualized_vectors = np.zeros(initial_pattern.shape + (9,)) # create a cell that will be used for visualizing vector
    visualized_vectors[:, :, 0] = initial_pattern

    #create a cell pattern for each angle
    bim1_img = Image.fromarray(initial_pattern) #  create a PIL (Pillow) image from a NumPy array, allow to perform rotation on img
    for angle_idx in range(0, 9):
        rotated = bim1_img.rotate(-angle_idx * 20, resample=Image.NEAREST)
        visualized_vectors[:, :, angle_idx] = np.array(rotated) / 255.0 #set back to numpy arrey from PIL img

    Y, X, Z = histogram.shape  # Z should be 9
    HOG_img = np.zeros((cell_size * Y, cell_size * X))

    for cell_y in range(Y):
        first_cell_index_y = cell_y * cell_size
        last_cell_index_y = (cell_y + 1) * cell_size
        for cell_x in range(X):
            first_cell_index_x = cell_x * cell_size
            last_cell_index_x = (cell_x + 1) * cell_size
            for bin_idx in range(9): #bin contains the rotated vectors (bin[0] - vector 0deg, bin[9] - vector 180deg)
                HOG_img[first_cell_index_y:last_cell_index_y, first_cell_index_x:last_cell_index_x] += visualized_vectors[:, :, bin_idx] * histogram[cell_y, cell_x, bin_idx]
    
    return HOG_img


def HOG(img):
    ###########################3
    #### GRADIENT
    ###########################3
    grad_img, orient_img = grad_rgb(img)


    orient_of_a_pixel = np.rad2deg(orient_img)
    # orient_of_a_pixel = np.mod(orient_of_a_pixel, 180)  # ensure all angles are in [0,180)

    ###########################3
    #### HISTOGRAM
    ###########################3
    cell_size = 8
    img_size_y = img.shape[0]
    img_size_x = img.shape[1]
    cells_num_y = np.int32(img_size_y/cell_size)
    cells_num_x = np.int32(img_size_x/cell_size)

    # bin container, we assume there is 9 bins (0, 20, 40.. deg up to 180)
    hist = np.zeros([cells_num_y, cells_num_x, 9], np.float32)

    # calculate histogram
    for j in range(0, cells_num_y):
        for i in range(0, cells_num_x):
            # extract cell from img
            cell_indx_start_y = j*cell_size
            cell_indx_end_y = (j+1)*cell_size
            cell_indx_start_x = i*cell_size
            cell_indx_end_x = (i+1)*cell_size

            cell_gradient_magnitude = grad_img[cell_indx_start_y:cell_indx_end_y, cell_indx_start_x:cell_indx_end_x]
            cell_orientation = orient_of_a_pixel[cell_indx_start_y:cell_indx_end_y, cell_indx_start_x:cell_indx_end_x]

            cell_gradient_magnitude = cell_gradient_magnitude.flatten()
            cell_orientation = cell_orientation.flatten()

            # histogram
            for k in range(0, cell_size*cell_size):
                pixel_magnitude = cell_gradient_magnitude[k]
                pixel_orientation = cell_orientation[k]

                if (pixel_orientation < 0):
                    pixel_orientation = pixel_orientation + 180

                # Wyliczenie przezdialu
                bin_centere = np.floor((pixel_orientation-10)/20)*20 +10 #  if the pixel’s orientation is, say, 37°, you want it to be assigned to the bin centered at 30

                #important if pixel_orientation is lass than 10
                if (bin_centere < 0):
                    bin_centere = 170

                bin_index = int((bin_centere-10)/20)
                next_bin_index = bin_index+1

                # wrapping
                if next_bin_index == 9:
                    next_bin_index=0

                # it might wrap around, thats why we choose smaller one
                d = min(abs(pixel_orientation-bin_centere), 180 - abs(pixel_orientation-bin_centere))/20

                # update histogram
                hist[j, i, bin_index] += + pixel_magnitude*(1-d)
                hist[j, i, next_bin_index] += + pixel_magnitude*(d)

    # Normalizacja w blokach
    e = math.pow(0.00001, 2)
    F_vector = []
    for j in range(0, cells_num_y-1):
        for i in range(0, cells_num_x-1):
            H0 = hist[j, i, :]
            H1 = hist[j, i+1, :]
            H2 = hist[j+1, i, :]
            H3 = hist[j+1, i+1, :]
            H = np.concatenate((H0, H1, H2, H3))
            n = np.linalg.norm(H)
            Hn = H/np.sqrt(math.pow(n, 2)+e)
            F_vector = np.concatenate((F_vector, Hn))

    print(len(F_vector))
    print(F_vector[0:10])

    return hist

img = cv2.imread('wiki/pos/per00060.ppm')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
hist = HOG(img)
imho = create_HOG_img(hist, 8)

#wyswietlanie obrazka
plt.subplot(1, 2, 1)
plt.imshow(imho, cmap='gray')
plt.title('HOG')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(img)
plt.title('Image ')
plt.axis('off')
plt.show()
