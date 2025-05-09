import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

def harris_response(image, filter_size=7, k=0.05):
    gray = np.float32(image)

    Ix = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=filter_size)
    Iy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=filter_size)

    Ixx = Ix * Ix
    Iyy = Iy * Iy
    Ixy = Ix * Iy

    Sxx = cv2.GaussianBlur(Ixx, (filter_size, filter_size), 0)
    Syy = cv2.GaussianBlur(Iyy, (filter_size, filter_size), 0)
    Sxy = cv2.GaussianBlur(Ixy, (filter_size, filter_size), 0)

    det_M = Sxx * Syy - Sxy * Sxy
    trace_M = Sxx + Syy
    H = det_M - k * (trace_M ** 2)

    H = cv2.normalize(H, None, 0, 1, cv2.NORM_MINMAX)
    return H

def find_max(image, size, threshold):
    data_max = ndimage.maximum_filter(image, size)
    maxima = (image == data_max)
    diff = image > threshold
    maxima[diff == 0] = 0
    return np.nonzero(maxima)

def draw_corners(image, x_coords, y_coords, title="Corners"):
    plt.figure()
    plt.imshow(image, cmap='gray')
    plt.plot(x_coords, y_coords, '*', color='red')
    plt.title(title)
    plt.axis('off')
    plt.show()

def process_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    H = harris_response(image, filter_size=7)
    y, x = find_max(H, size=7, threshold=0.4)
    draw_corners(image, x, y, title=f"Corners in {image_path}")

def get_descriptors(image, coords, size):
    h, w = image.shape
    pts = list(filter(lambda pt: pt[0] >= size and pt[0] < h - size and pt[1] >= size and pt[1] < w - size, zip(coords[0], coords[1])))
    
    patches = []
    for y, x in pts:
        patch = image[y - size:y + size + 1, x - size:x + size + 1]
        patches.append(patch.flatten())
    return list(zip(patches, pts))

def match_descriptors(desc1, desc2, n_matches=20):
    matches = []
    for vec1, pt1 in desc1:
        best_score = float('inf')
        best_match = None
        for vec2, pt2 in desc2:
            score = np.sum(np.abs(vec1 - vec2))
            if score < best_score:
                best_score = score
                best_match = (pt1, pt2, score)
        matches.append(best_match)
    
    matches.sort(key=lambda x: x[2])
    return matches[:n_matches]

def get_normalized_descriptors(image, coords, size):
    h, w = image.shape
    pts = list(filter(lambda pt: pt[0] >= size and pt[0] < h - size and pt[1] >= size and pt[1] < w - size, zip(coords[0], coords[1])))

    patches = []
    for y, x in pts:
        patch = image[y - size:y + size + 1, x - size:x + size + 1].astype(np.float32)
        mean = np.mean(patch)
        std = np.std(patch)
        if std > 0:
            patch = (patch - mean) / std
        else:
            patch = patch * 0
        patches.append(patch.flatten())
    return list(zip(patches, pts))

def plot_matches(img1, img2, matches):
    h1, w1 = img1.shape
    h2, w2 = img2.shape
    out_img = np.zeros((max(h1, h2), w1 + w2), dtype=np.uint8)
    out_img[:h1, :w1] = img1
    out_img[:h2, w1:] = img2

    plt.figure(figsize=(10, 6))
    plt.imshow(out_img, cmap='gray')
    for (y1, x1), (y2, x2), _ in matches:
        plt.plot([x1, x2 + w1], [y1, y2], 'r-', linewidth=0.8)
        plt.plot(x1, y1, 'go')
        plt.plot(x2 + w1, y2, 'bo')
    plt.axis('off')
    plt.show()

def full_pipeline(img_path1, img_path2, use_normalized=False, patch_size=15, n_matches=20):
    img1 = cv2.imread(img_path1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img_path2, cv2.IMREAD_GRAYSCALE)

    harris1 = harris_response(img1)
    harris2 = harris_response(img2)

    coords1 = find_max(harris1, size=7, threshold=0.4)
    coords2 = find_max(harris2, size=7, threshold=0.4)

    if use_normalized:
        desc1 = get_normalized_descriptors(img1, coords1, patch_size)
        desc2 = get_normalized_descriptors(img2, coords2, patch_size)
    else:
        desc1 = get_descriptors(img1, coords1, patch_size)
        desc2 = get_descriptors(img2, coords2, patch_size)

    matches = match_descriptors(desc1, desc2, n_matches=n_matches)
    plot_matches(img1, img2, matches)

img1 = cv2.imread('fontanna1.jpg', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('fontanna2.jpg', cv2.IMREAD_GRAYSCALE)

def fast_with_harris_score(img):
    fast = cv2.FastFeatureDetector_create(threshold=10, nonmaxSuppression=False)
    keypoints = fast.detect(img, None)

    harris = cv2.cornerHarris(np.float32(img), 2, 3, 0.04)
    harris = cv2.dilate(harris, None)

    points = []
    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        if 0 <= y < harris.shape[0] and 0 <= x < harris.shape[1]:
            points.append((x, y, harris[y, x]))
    return points

def nms(points, size=3):
    output = []
    points = sorted(points, key=lambda p: -p[2])
    used = np.zeros((10000, 10000), dtype=bool)
    for x, y, score in points:
        if not used[y, x]:
            output.append((x, y, score))
            used[y - size//2:y + size//2 + 1, x - size//2:x + size//2 + 1] = True
    return output

def filter_border(points, img_shape, patch_size=31):
    margin = patch_size // 2
    h, w = img_shape
    return [(x, y, score) for (x, y, score) in points if margin <= x < w - margin and margin <= y < h - margin]

def select_best(points, N=200):
    return sorted(points, key=lambda p: -p[2])[:N]

def compute_orientation(img, x, y, patch_size=31):
    patch = img[y - patch_size//2:y + patch_size//2 + 1, x - patch_size//2:x + patch_size//2 + 1]
    m = cv2.moments(patch)
    if m["m00"] == 0:
        return 0
    cx = m["m10"] / m["m00"]
    cy = m["m01"] / m["m00"]
    dx = cx - patch_size // 2
    dy = cy - patch_size // 2
    angle = np.arctan2(dy, dx)
    return angle

def brief_descriptor(img, points, pairs, patch_size=31):
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    descriptors = []

    h, w = img.shape
    margin = patch_size // 2

    for (x, y, score) in points:
        angle = compute_orientation(blur, x, y, patch_size)
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)

        desc = []
        valid = True

        for (dx1, dy1, dx2, dy2) in pairs:
            rx1 = int(cos_a * dx1 - sin_a * dy1)
            ry1 = int(sin_a * dx1 + cos_a * dy1)
            rx2 = int(cos_a * dx2 - sin_a * dy2)
            ry2 = int(sin_a * dx2 + cos_a * dy2)

            x1 = x + rx1
            y1 = y + ry1
            x2 = x + rx2
            y2 = y + ry2

            if not (0 <= x1 < w and 0 <= y1 < h and 0 <= x2 < w and 0 <= y2 < h):
                valid = False
                break

            p1 = blur[y1, x1]
            p2 = blur[y2, x2]
            desc.append(1 if p1 < p2 else 0)

        if valid:
            descriptors.append((desc, (x, y)))

    return descriptors

def load_orb_pairs(filename='orb_descriptor_positions.txt'):
    return np.loadtxt(filename).astype(int)

def generate_random_pairs(n=256, patch_radius=15):
    np.random.seed(42)
    return [(
        np.random.randint(-patch_radius, patch_radius + 1),
        np.random.randint(-patch_radius, patch_radius + 1),
        np.random.randint(-patch_radius, patch_radius + 1),
        np.random.randint(-patch_radius, patch_radius + 1)
    ) for _ in range(n)]

def hamming_distance(desc1, desc2):
    return sum(a != b for a, b in zip(desc1, desc2))

def match_orb(descs1, descs2, N=20):
    matches = []
    for d1, pt1 in descs1:
        best_score = 9999
        best_pt = None
        for d2, pt2 in descs2:
            score = hamming_distance(d1, d2)
            if score < best_score:
                best_score = score
                best_pt = pt2
        matches.append((pt1, best_pt, best_score))
    matches.sort(key=lambda x: x[2])
    return matches[:N]

def plot_matches2(img1, img2, matches):
    h1, w1 = img1.shape
    h2, w2 = img2.shape
    canvas = np.zeros((max(h1, h2), w1 + w2), dtype=np.uint8)
    canvas[:h1, :w1] = img1
    canvas[:h2, w1:] = img2

    plt.figure(figsize=(10, 6))
    plt.imshow(canvas, cmap='gray')
    for (x1, y1), (x2, y2), _ in matches:
        plt.plot([x1, x2 + w1], [y1, y2], 'r-')
        plt.plot(x1, y1, 'go')
        plt.plot(x2 + w1, y2, 'bo')
    plt.axis('off')
    plt.show()

def run_orb_pipeline(img1_path, img2_path):
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

    pts1 = fast_with_harris_score(img1)
    pts2 = fast_with_harris_score(img2)

    pts1 = filter_border(nms(pts1), img1.shape)
    pts2 = filter_border(nms(pts2), img2.shape)

    pts1 = select_best(pts1, 200)
    pts2 = select_best(pts2, 200)

    pairs = generate_random_pairs()

    desc1 = brief_descriptor(img1, pts1, pairs)
    desc2 = brief_descriptor(img2, pts2, pairs)

    matches = match_orb(desc1, desc2)
    plot_matches2(img1, img2, matches)

def stitch_images(left_path, right_path, feature_detector='ORB', use_knn=True, good_match_ratio=0.5):
    img_left = cv2.imread(left_path)
    img_right = cv2.imread(right_path)
    gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

    if feature_detector == 'SIFT':
        detector = cv2.SIFT_create()
        norm = cv2.NORM_L2
    elif feature_detector == 'SURF':
        detector = cv2.xfeatures2d.SURF_create()
        norm = cv2.NORM_L2
    else:
        detector = cv2.ORB_create(5000)
        norm = cv2.NORM_HAMMING

    kp1, desc1 = detector.detectAndCompute(gray_left, None)
    kp2, desc2 = detector.detectAndCompute(gray_right, None)

    img_kp1 = cv2.drawKeypoints(img_left, kp1, None, color=(0, 255, 0))
    img_kp2 = cv2.drawKeypoints(img_right, kp2, None, color=(0, 255, 0))
    plt.figure(figsize=(12,6))
    plt.subplot(121), plt.imshow(img_kp1[..., ::-1]), plt.title('Left Keypoints')
    plt.subplot(122), plt.imshow(img_kp2[..., ::-1]), plt.title('Right Keypoints')
    plt.show()

    bf = cv2.BFMatcher(norm, crossCheck=not use_knn)

    if use_knn:
        matches = bf.knnMatch(desc1, desc2, k=2)
        best_matches = [[m] for m, n in matches if m.distance < good_match_ratio * n.distance]
        flat_matches = [m[0] for m in best_matches]
    else:
        matches = bf.match(desc1, desc2)
        matches = sorted(matches, key=lambda x: x.distance)
        flat_matches = matches[:50]

    matched_img = cv2.drawMatches(img_left, kp1, img_right, kp2, flat_matches, None, flags=2)
    plt.figure(figsize=(15,7))
    plt.imshow(matched_img[..., ::-1])
    plt.title("Matched Points")
    plt.show()

    ptsA = np.float32([kp1[m.queryIdx].pt for m in flat_matches])
    ptsB = np.float32([kp2[m.trainIdx].pt for m in flat_matches])

    H, mask = cv2.findHomography(ptsA, ptsB, cv2.RANSAC)

    height = max(img_left.shape[0], img_right.shape[0])
    width = img_left.shape[1] + img_right.shape[1]
    result = cv2.warpPerspective(img_left, H, (width, height))
    result[0:img_right.shape[0], 0:img_right.shape[1]] = img_right

    gray_result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray_result, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(contours[0])
    final_result = result[y:y+h, x:x+w]

    plt.figure(figsize=(12,8))
    plt.imshow(final_result[..., ::-1])
    plt.title("Final Panorama")
    plt.axis('off')
    plt.show()

def ex1():
    image_paths = ["fontanna1.jpg", "fontanna2.jpg", "budynek1.jpg", "budynek2.jpg"]
    for path in image_paths:
        process_image(path)

def ex2():
    full_pipeline("fontanna1.jpg", "fontanna2.jpg")
    full_pipeline("budynek1.jpg", "budynek2.jpg")
    full_pipeline("fontanna1.jpg", "fontanna_pow.jpg")
    full_pipeline("eiffel1.jpg", "eiffel2.jpg")

    full_pipeline("eiffel1.jpg", "eiffel2.jpg", use_normalized=True)

def ex3():
    run_orb_pipeline("C:\\Users\\kajtek\\Desktop\\studia_2025\\advanced_vision_systems\\zaw_avs_materials\\lab06_fp\\fontanna1.jpg", "C:\\Users\\kajtek\\Desktop\\studia_2025\\advanced_vision_systems\\zaw_avs_materials\\lab06_fp\\fontanna2.jpg")
    # run_orb_pipeline("budynek1.jpg", "budynek2.jpg")
    # run_orb_pipeline("fontanna1.jpg", "fontanna_pow.jpg")
    # run_orb_pipeline("eiffel1.jpg", "eiffel2.jpg")

def ex4():
    stitch_images("left_panorama.jpg", "right_panorama.jpg", feature_detector='ORB')

# ex1()
# ex2()
ex3()
#ex4()