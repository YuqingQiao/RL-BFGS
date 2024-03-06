import os
import cv2 as cv
import numpy as np

from matplotlib import pyplot as plt
import time

from camera_librealsense import Camera



BLUE = [(0, 35, 30), (30, 70, 150)]
RED = [(50, 0, 0), (150, 30, 30)]

CHESSBOARD_SIZE = (7, 6)
SQUARE_SIZE = 25  # [mm]

cam = Camera()
cam.start()

#cam.plot_frames()

# Take some pictures from different orientations and positions
take_pictures = 1

calibrate = 1

if take_pictures:
    i = 0
    #while True:
        #input("Press Enter to capture a frame...")
    color_frame, depth_frame = cam.get_frames()

    color_filename = os.path.join("./calibration/photos", f"color_frame_{i}.png")
    cv.imwrite(color_filename, color_frame)
    depth_filename = os.path.join("./calibration/photos", f"depth_frame_{i}.png")
    cv.imwrite(depth_filename, depth_frame)

    i += 1
    plt.imshow(color_frame)
    plt.show()
        # simply quit process after enough pictures

# Now get the calibration matrices
# Prepare object points

if calibrate:
    objp = np.zeros((CHESSBOARD_SIZE[1]*CHESSBOARD_SIZE[0], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2) * SQUARE_SIZE

    # Store object and image points from all images
    objpoints = []
    imgpoints = []

    # Load and process each image
    dir = "./calibration/photos"
    for file in os.listdir(dir):
        img = cv.imread(os.path.join(dir, file))
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # Find chessboard corners
        ret, corners = cv.findChessboardCorners(gray, CHESSBOARD_SIZE, None)

        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)

    # Calibrate camera
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    # Check reprojeciton error
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
        mean_error += error
    print("total error: {}".format(mean_error / len(objpoints)))

    np.save("./calibration/mtx", mtx)
    np.save("./calibration/dist", dist)

check_position = False
if check_position:
    while True:
        cam.plot_frames()
        # body_pos = cam.get_body_pos_cv()
        time.sleep(2)
