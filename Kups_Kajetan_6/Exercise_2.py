import numpy as np
import cv2
from pathlib import Path

import matplotlib.pyplot as plt

#####################################################################3
########################  import files path
#####################################################################3
parent_dir = Path(__file__).resolve().parent.parent
path = parent_dir.as_posix()
image_dir = path +  "/zaw_avs_materials/" + "lab05_stereo/" + "pairs/"
# image_dir = parent_dir / "zaw_avs_materials" / "lab05_stereo" / "aloes" 


# def read_frame(name, scale_percent):
#     image_path = image_dir /name
#     I = cv2.imread(image_path)
#     I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
    
#     # Calculate the new dimensions
#     width = int(I.shape[1] * scale_percent / 100)
#     height = int(I.shape[0] * scale_percent / 100)
#     dim = (width, height)

#     return cv2.resize(I, dim, interpolation=cv2.INTER_AREA)


def calibrate_camera():
     # termination criteria
    criteria = ( cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_FIX_SKEW

    # inner size of chessboard
    width = 9
    height = 6
    square_size = 0.025 # 0.025 meters

    # prepare object points , like (0 ,0 ,0) , (1 ,0 ,0) , (2 ,0 ,0) .... ,(8 ,6 ,0)
    objp = np . zeros (( height * width , 1 , 3) , np . float64 )
    objp [: , 0, :2] = np . mgrid [0: width , 0: height ]. T. reshape ( -1 , 2)

    objp = objp * square_size # Create real world coords . Use your metric .

    # Arrays to store object points and image points from all the images .
    objpoints = [] # 3d point in real world space
    imgpoints_left = []  # 2d points in left image
    imgpoints_right = [] # 2d points in right image

    img_width = 640
    img_height = 480
    image_size = ( img_width, img_height )
    number_of_images = 50

    for i in range(1, number_of_images):
        # read image
        img_left = cv2.imread(image_dir + f"left_{i:02d}.png")
        img_right = cv2.imread(image_dir + f"right_{i:02d}.png")

        gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret_left, corners_left = cv2.findChessboardCorners(gray_left, (width, height), 
                        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
        ret_right, corners_right = cv2.findChessboardCorners(gray_right, (width, height), 
                    cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
        
        Y , X , channels = img_left.shape
    
        # skip images where the corners of the chessboard are too close to the edges of the image
        x_thresh_bad = None
        y_thresh_bad = None
        if (ret_left):
            minRx = corners_left[:, :, 0].min()
            maxRx = corners_left[:, :, 0].max()
            minRy = corners_left[:, :, 1].min()
            maxRy = corners_left[:, :, 1].max()
            border_threshold_x = X /12
            border_threshold_y = Y /12
            x_thresh_bad = False
        
        if ( minRx < border_threshold_x ) :
            x_thresh_bad = True
            y_thresh_bad = False
        
        if ( minRy < border_threshold_y ) :
            y_thresh_bad = True

        if ( y_thresh_bad == True ) or ( x_thresh_bad == True ):
            continue

        # If found, add object points, image points ( after refining them )
        if ret_left and ret_right :

            # improving the location of points (sub - pixel )
            corners_left = cv2.cornerSubPix(gray_left, corners_left, (3, 3), (-1, -1), criteria)
            corners_right = cv2.cornerSubPix(gray_right, corners_right, (3, 3), (-1, -1), criteria)

            objpoints.append(objp)
            imgpoints_left.append(corners_left)
            imgpoints_right.append(corners_right)
            
            # Draw corners for visualization
            cv2.drawChessboardCorners(img_left, (width, height), corners_left, ret_left)
            cv2.drawChessboardCorners(img_right, (width, height), corners_right, ret_right)
            cv2.imshow("Left Corners", img_left)
            cv2.imshow("Right Corners", img_right)
            cv2.waitKey(5)
        else:
            print (" Chessboard couldn ’t detected . Image pair : ", i)
            continue

##########################################################################################
##########################################################################################

    N_OK = len( objpoints )
    K_left = np.zeros((3, 3))
    D_left = np.zeros((4, 1))
    K_right = np.zeros((3, 3))
    D_right = np.zeros((4, 1))
    rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in range(N_OK)]
    tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in range(N_OK)]

    ret, K_left, D_left, _, _ = \
        cv2 . fisheye.calibrate(
            objpoints,
            imgpoints_left,
            image_size,
            K_left,
            D_left,
            rvecs,
            tvecs,
            calibration_flags,
            ( cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
        )

    ret, K_right, D_right, _, _ = \
        cv2 . fisheye.calibrate(
            objpoints,
            imgpoints_right,
            image_size,
            K_right,
            D_right,
            rvecs,
            tvecs,
            calibration_flags,
            ( cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
        )

    return K_left, D_left, K_right, D_right, objpoints, imgpoints_left, imgpoints_right, image_size



if __name__ == "__main__":
    K_left, D_left, K_right, D_right, objpoints, imgpoints_left, imgpoints_right, image_size = calibrate_camera()

    imgpointsLeft = np.asarray(imgpoints_left, dtype=np.float64 )
    imgpointsRight = np.asarray(imgpoints_right, dtype=np.float64 )
    (RMS , _ , _ , _ , _ , rotationMatrix, translationVector) = cv2.fisheye.stereoCalibrate (
            objpoints, imgpointsLeft, imgpointsRight,
            K_left, D_left,
            K_right, D_right,
            image_size, None, None,
            cv2.CALIB_FIX_INTRINSIC,
            ( cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01) )
    R2 = np . zeros ([3 ,3])
    P1 = np . zeros ([3 ,4])
    P2 = np . zeros ([3 ,4])
    Q = np . zeros ([4 ,4])

    # Rectify calibration results
    (leftRectification, rightRectification, leftProjection, rightProjection, dispartityToDepthMap ) = cv2.fisheye.stereoRectify(
                    K_left, D_left,
                    K_right, D_right,
                    image_size,
                    rotationMatrix, translationVector,
                    0, R2, P1, P2, Q,
                    cv2.CALIB_ZERO_DISPARITY , (0 ,0) , 0 , 0)
    
    map1_left, map2_left = cv2.fisheye.initUndistortRectifyMap( K_left, D_left, leftRectification, leftProjection, image_size, cv2.CV_16SC2 )
    map1_right, map2_right = cv2.fisheye.initUndistortRectifyMap(K_right, D_right, rightRectification, rightProjection, image_size, cv2.CV_16SC2)


    img_num = 15
    image_L = cv2.imread( image_dir + f"left_{img_num:02d}.png")
    image_R = cv2.imread( image_dir + f"right_{img_num:02d}.png")

    dst_L = cv2.remap(image_L, map1_left, map2_left, cv2.INTER_LINEAR)
    dst_R = cv2.remap(image_R, map1_right, map2_right, cv2.INTER_LINEAR)

    # cv2.imshow ("original image left", image_L)
    # cv2.imshow ("undistorted_image left", dst_L)
    # cv2.imshow ("original image right", image_R)
    # cv2.imshow ("undistorted_image right", dst_R)

    N, XX, YY = dst_L.shape[:: -1] # RGB image size
    visRectify = np.zeros(( YY, XX*2, N), np.uint8)
    visRectify [: ,0: XX : , :] = dst_L # left image assignment
    visRectify [: , XX : XX *2: ,:] = dst_R # right image assignment
    # draw horizontal lines
    for y in range (0 , YY ,10) :
        cv2.line( visRectify , (0 , y) , ( XX *2 , y) , (255 ,0 ,0) )
        cv2.imshow('visRectify', visRectify ) # display image with lines
    cv2.waitKey()