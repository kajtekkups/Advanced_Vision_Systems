import numpy as np
import cv2
import imutils
import os
from os.path import join
from pathlib import Path

current_dir = Path.cwd()
DATASET_DIR = current_dir / 'Kups_Kajetan_8' / 'sequences'

SIGMA = 17
SEARCH_REGION_SCALE = 2
LR = 0.1
NUM_PRETRAIN = 256
VISUALIZE = True
TEMPLATE_SIZE = (120, 120)  # Fixed size for all patches

def load_gt(gt_file):
    with open(gt_file, 'r') as file:
        lines = file.readlines()
    lines = [line.split(',') for line in lines]
    lines = [[int(float(coord)) for coord in line] for line in lines]
    return lines

def crop_search_window(bbox, frame):
    xmin, ymin, width, height = bbox
    center_x = xmin + width / 2
    center_y = ymin + height / 2

    search_width = width * SEARCH_REGION_SCALE
    search_height = height * SEARCH_REGION_SCALE

    xmin = int(center_x - search_width / 2)
    ymin = int(center_y - search_height / 2)
    xmax = int(center_x + search_width / 2)
    ymax = int(center_y + search_height / 2)

    y_pad = int(search_height)
    x_pad = int(search_width)
    padded_frame = cv2.copyMakeBorder(frame, y_pad, y_pad, x_pad, x_pad, cv2.BORDER_REFLECT)

    xmin += x_pad
    xmax += x_pad
    ymin += y_pad
    ymax += y_pad

    window = padded_frame[ymin:ymax, xmin:xmax]
    window = cv2.cvtColor(window, cv2.COLOR_BGR2GRAY)

    # cv2.imshow('crop', window)


    window = cv2.resize(window, TEMPLATE_SIZE)
    
    # cv2.waitKey(2)
    return window

def pre_process(img):
    height, width = img.shape
    img = img.astype(np.float32)
    img = np.log(img + 1.0)
    img = (img - np.mean(img)) / (np.std(img) + 1e-5)

    win_col = np.hanning(width)
    win_row = np.hanning(height)
    mask_col, mask_row = np.meshgrid(win_col, win_row)
    window = mask_col * mask_row
    img = img * window
    return img


def get_gauss_response(gt_box):

    width = int(gt_box[2] * SEARCH_REGION_SCALE)
    height = int(gt_box[3] * SEARCH_REGION_SCALE)
    xx, yy = np.meshgrid(np.arange(width), np.arange(height))

    center_x = width // 2
    center_y = height // 2
    dist = (np.square(xx - center_x) + np.square(yy - center_y)) / (2 * SIGMA)
    response = np.exp(-dist)

    response = cv2.resize(response, TEMPLATE_SIZE)
    return response

def random_warp(img):
    random_angle = np.random.uniform(-5, 5)
    img_rot = imutils.rotate_bound(img, random_angle)
    img_resized = cv2.resize(img_rot, (img.shape[1], img.shape[0]))

    # cv2.imshow('original', img)
    # cv2.imshow('Random Warp', img_resized)
    # cv2.waitKey(3)  # Wait for any key to close the window

    return img_resized

def pre_training(init_gt, init_frame, G):
    template = crop_search_window(init_gt, init_frame)
    fi = pre_process(template)

    # cv2.imshow('Random Warp', fi)
    # cv2.waitKey(3)

    Ai = G * np.conjugate(np.fft.fft2(fi))
    Bi = np.fft.fft2(fi) * np.conjugate(np.fft.fft2(fi))

    for _ in range(NUM_PRETRAIN):
        fi = pre_process(random_warp(template))
        Ai += G * np.conjugate(np.fft.fft2(fi))
        Bi += np.fft.fft2(fi) * np.conjugate(np.fft.fft2(fi))

        # cv2.imshow('Random Warp', fi)
        # cv2.waitKey(3)
    return Ai, Bi

def initialize(init_frame, init_gt):
    g = get_gauss_response(init_gt)
    G = np.fft.fft2(g)
    
    Ai, Bi = pre_training(init_gt, init_frame, G)
    return Ai, Bi, G

def predict(frame, position, H):
    patch = crop_search_window(position, frame)
    
    patch = pre_process(patch)
    cv2.imshow("predict", patch)
    cv2.waitKey(3)

    F = np.fft.fft2(patch)
    gi = np.fft.ifft2(H * F)
    return np.real(gi)

def update(frame, position, Ai, Bi, G):
    patch = crop_search_window(position, frame)
    patch = pre_process(patch)
    F = np.fft.fft2(patch)

    Ai = (1 - LR) * Ai + LR * (G * F.conj())
    Bi = (1 - LR) * Bi + LR * (F * F.conj())

    return Ai, Bi

def update_position(spatial_response, position):
    h, w = spatial_response.shape
    max_idx = np.argmax(spatial_response)
    dy, dx = np.unravel_index(max_idx, spatial_response.shape)

    cv2.imshow("spatial_response", spatial_response)
    cv2.waitKey(3)

    center_y, center_x = h // 2, w // 2
    shift_y = dy - center_y
    shift_x = dx - center_x

    # Clamp the shift to avoid large jumps
    MAX_SHIFT = 15
    shift_x = np.clip(shift_x, -MAX_SHIFT, MAX_SHIFT)
    shift_y = np.clip(shift_y, -MAX_SHIFT, MAX_SHIFT)

    x, y, width, height = position
    new_x = x + shift_x
    new_y = y + shift_y

    return [new_x, new_y, width, height]

def track(image, position, Ai, Bi, G):
    H = Ai/Bi
    response = predict(image, position, H)
    new_position = update_position(response, position)
    newAi, newBi = update(image, new_position, Ai, Bi, G)
    new_position = [int(x) for x in new_position]
    return new_position, newAi, newBi

def bbox_iou(box1, box2):
    b1_x1, b1_x2 = box1[0], box1[0] + box1[2]
    b1_y1, b1_y2 = box1[1], box1[1] + box1[3]
    b2_x1, b2_x2 = box2[0], box2[0] + box2[2]
    b2_y1, b2_y2 = box2[1], box2[1] + box2[3]

    inter_rect_x1 = max(b1_x1, b2_x1)
    inter_rect_y1 = max(b1_y1, b2_y1)
    inter_rect_x2 = min(b1_x2, b2_x2)
    inter_rect_y2 = min(b1_y2, b2_y2)

    inter_area = np.clip(inter_rect_x2 - inter_rect_x1, 0, None) * np.clip(inter_rect_y2 - inter_rect_y1, 0, None)
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)
    return iou

def test_sequence(DATASET_DIR, sequence):
    seqdir = join(DATASET_DIR, sequence)
    imgdir = join(seqdir, 'color')
    imgnames = os.listdir(imgdir)
    imgnames.sort()

    init_img = cv2.imread(join(imgdir, imgnames[0]))
    gt_boxes = load_gt(join(seqdir, 'groundtruth.txt'))
    position = gt_boxes[0]
    Ai, Bi, G = initialize(init_img, position)

    if VISUALIZE:
        cv2.rectangle(init_img, (position[0], position[1]), (position[0] + position[2], position[1] + position[3]), (255, 0, 0), 2)
        cv2.imshow('demo', init_img)
        cv2.waitKey(2)

    results = []
    for imgname in imgnames[1:]:
        img = cv2.imread(join(imgdir, imgname))
        position, Ai, Bi = track(img, position, Ai, Bi, G)
        results.append(position.copy())

        if VISUALIZE:
            cv2.rectangle(img, (position[0], position[1]), (position[0] + position[2], position[1] + position[3]), (255, 0, 0), 2)
            cv2.imshow('demo', img)
            cv2.waitKey(2)

    cv2.destroyAllWindows()
    return results, gt_boxes


def show_sequence_2(sequence_dir):

    imgdir = join(sequence_dir, 'color')
    imgnames = os.listdir(imgdir)                  
    imgnames.sort()
    gt_boxes = load_gt(join(sequence_dir, 'groundtruth.txt'))

    for imgname, gt in zip(imgnames, gt_boxes):
        img = cv2.imread(join(imgdir, imgname))
        position = [int(x) for x in gt]
        cv2.rectangle(img, (position[0], position[1]), (position[0]+position[2], position[1]+position[3]), (255, 0, 0), 2)
        cv2.imshow('demo', img)
            
        window = crop_search_window(position, img)
        cv2.imshow('search window', window.astype(np.uint8))
        cv2.waitKey(10)

    cv2.destroyAllWindows()
# show_sequence_2(join(DATASET_DIR, 'jump'))        


# Run your test sequence here:
sequences = ['jump', 'sunshade']
ious_per_sequence = {}

for sequence in sequences:
    results, gt_boxes = test_sequence(DATASET_DIR, sequence)
    ious = []
    for res_box, gt_box in zip(results, gt_boxes[1:]):
        iou = bbox_iou(res_box, gt_box)
        ious.append(iou)
    ious_per_sequence[sequence] = np.mean(ious)
    print(sequence, ':', np.mean(ious))

print('Mean IoU:', np.mean(list(ious_per_sequence.values())))
