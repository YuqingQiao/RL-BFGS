#from imutils.video import VideoStream
import argparse
import imutils
import time
import cv2
import sys
from camera_librealsense import Camera
import numpy as np

#measuring distance between aruco markers in stream with rs-camera.
##measuring velocity of a single marker in 3d-space.
##make sure markers are in front of the camera. simply run this file to start.

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--type", type=str,
	default="DICT_APRILTAG_36h11",
	help="type of ArUCo tag to detect")
args = vars(ap.parse_args())
# define names of each possible ArUco tag OpenCV supports
ARUCO_DICT = {
	"DICT_4X4_50": cv2.aruco.DICT_4X4_50,
	"DICT_4X4_100": cv2.aruco.DICT_4X4_100,
	"DICT_4X4_250": cv2.aruco.DICT_4X4_250,
	"DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
	"DICT_5X5_50": cv2.aruco.DICT_5X5_50,
	"DICT_5X5_100": cv2.aruco.DICT_5X5_100,
	"DICT_5X5_250": cv2.aruco.DICT_5X5_250,
	"DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
	"DICT_6X6_50": cv2.aruco.DICT_6X6_50,
	"DICT_6X6_100": cv2.aruco.DICT_6X6_100,
	"DICT_6X6_250": cv2.aruco.DICT_6X6_250,
	"DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
	"DICT_7X7_50": cv2.aruco.DICT_7X7_50,
	"DICT_7X7_100": cv2.aruco.DICT_7X7_100,
	"DICT_7X7_250": cv2.aruco.DICT_7X7_250,
	"DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
	"DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
	"DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
	"DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
	"DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
	"DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
}
# verify that the supplied ArUCo tag exists and is supported by
# OpenCV
if ARUCO_DICT.get(args["type"], None) is None:
	print("[INFO] ArUCo tag of '{}' is not supported".format(
		args["type"]))
	sys.exit(0)
# load the ArUCo dictionary and grab the ArUCo parameters
print("[INFO] detecting '{}' tags...".format(args["type"]))
arucoDict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[args["type"]])
arucoParams = cv2.aruco.DetectorParameters()
# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
cam = Camera()
cam.start()

last_pos = [-1, -1]
current_pos = [-1, -1]
last_depth = 0
current_depth = 0
length_per_pixel = 0

time.sleep(1.0)
# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 1000 pixels
	color_frame, depth_frame = cam.get_frames()
	#frame = imutils.resize(frame, width=1000)
	# detect ArUco markers in the input frame
	detector = cv2.aruco.ArucoDetector(arucoDict, arucoParams)
	(corners, ids, rejected) = detector.detectMarkers(color_frame)
	# verify *at least* one ArUco marker was detected
	if len(corners) > 0:
		# flatten the ArUco IDs list
		centers = []
		ids = ids.flatten()

		# loop over the detected ArUCo corners
		for (markerCorner, markerID) in zip(corners, ids):
			# extract the marker corners (which are always returned
			# in top-left, top-right, bottom-right, and bottom-left
			# order)
			corners = markerCorner.reshape((4, 2))
			(topLeft, topRight, bottomRight, bottomLeft) = corners
			# convert each of the (x, y)-coordinate pairs to integers
			topRight = (int(topRight[0]), int(topRight[1]))
			bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
			bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
			topLeft = (int(topLeft[0]), int(topLeft[1]))
			# draw the bounding box of the ArUCo detection
			cv2.line(color_frame, topLeft, topRight, (0, 255, 0), 2)
			cv2.line(color_frame, topRight, bottomRight, (0, 255, 0), 2)
			cv2.line(color_frame, bottomRight, bottomLeft, (0, 255, 0), 2)
			cv2.line(color_frame, bottomLeft, topLeft, (0, 255, 0), 2)
			# compute and draw the center (x, y)-coordinates of the
			# ArUco marker
			cX = int((topLeft[0] + bottomRight[0]) / 2.0)
			cY = int((topLeft[1] + bottomRight[1]) / 2.0)
			cv2.circle(color_frame, (cX, cY), 4, (0, 0, 255), -1)

			centers.append([cX, cY])

			# draw the ArUco marker ID on the color_frame
			cv2.putText(color_frame, str(markerID),
						(topLeft[0], topLeft[1] - 15),
						cv2.FONT_HERSHEY_SIMPLEX,
						0.5, (0, 255, 0), 2)

		ids = list(ids)

		#make sure the programme continue when detection fails
		try:
			idx_00 = ids.index(2)
			idx_01 = ids.index(3)
			idx_02 = ids.index(4)
			idx_03 = ids.index(5)
		except ValueError:
			last_pos = [-1, -1]
			continue

		dist_bottom = np.linalg.norm(np.array(centers[idx_00]) - np.array(centers[idx_01]), ord=2)

		dist_top = np.linalg.norm(np.array(centers[idx_02]) - np.array(centers[idx_03]), ord=2)

		dists = [dist_bottom, dist_top]
		indexs = [idx_00, idx_02]
		real_distances = []
		for i in range(len(indexs)):
			(topLeft, topRight, bottomRight, bottomLeft) = corners

			pixel_len = (np.linalg.norm(topLeft - topRight, ord=2) + np.linalg.norm(bottomLeft - bottomRight, ord=2)) / 2
			length_per_pixel = 0.04 / pixel_len
			real_distances.append(np.round(dists[i] * length_per_pixel, 4))

		ids = [idx_00, idx_01, idx_02, idx_03]
		for i in range(len(ids) // 2):
			cv2.line(color_frame, centers[ids[2 * i]], centers[ids[2 * i + 1]], (255, 0, 0), 2)
			cv2.putText(color_frame, str(real_distances[i]),
						(int((centers[ids[2 * i]][0] + centers[ids[2 * i + 1]][0]) / 2), int((centers[ids[2 * i]][1] + centers[ids[2 * i + 1]][1]) / 2)),
						cv2.FONT_HERSHEY_SIMPLEX, 0.5,
						(0, 0, 255), 2)

		#measuring velocity
		if (last_pos[0] == -1):
			last_pos = centers[idx_00]
			last_depth = depth_frame[last_pos[1], last_pos[0]]
		else:
			# cal vel
			current_pos = centers[idx_00]
			current_depth = depth_frame[current_pos[1], current_pos[0]]
			vel_x = current_pos[0] - last_pos[0]
			vel_y = current_pos[1] - last_pos[1]
			vel_z = np.round((current_depth/1000 - last_depth/1000), 5)
			vel_xy = np.round(np.linalg.norm([vel_x, vel_y], ord=2), 5)
			real_vel_xy = np.round(vel_xy * length_per_pixel, 5) * 30
			real_vel = np.round(np.linalg.norm([real_vel_xy, vel_z], ord=2) * 30, 5) * 30
			# arrow visualization
			end_point = (10*vel_x + current_pos[0], 10*vel_y + current_pos[1])

			cv2.arrowedLine(color_frame, centers[idx_00], end_point, (238, 138, 238), 3)

			cv2.putText(color_frame, str(real_vel_xy),
						(100, 100),
						cv2.FONT_HERSHEY_SIMPLEX, 0.5,
						(0, 0, 255), 2)
			cv2.putText(color_frame, str(current_depth),
						(100, 120),
						cv2.FONT_HERSHEY_SIMPLEX, 0.5,
						(0, 0, 255), 2)

			last_pos = current_pos
			last_depth = current_depth

	# show the output frame
	cv2.imshow("Frame", color_frame)
	key = cv2.waitKey(1) & 0xFF
		# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break
	# do a bit of cleanup
cv2.destroyAllWindows()
cam.stop()