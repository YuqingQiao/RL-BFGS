import argparse
import imutils
import sys
import cv2 as cv

#testing aruco markers in image with rs-camera. be sure the image is in the path. E.g. use --image calibration/photos/color_frame_0.png
# simply run this file to start.

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,

                help="path to input image containing ArUCo tag")
ap.add_argument("-t", "--type", type=str,
                default="DICT_APRILTAG_36h11",
                help="type of ArUCo tag to detect")
args = vars(ap.parse_args())
ARUCO_DICT = {
    "DICT_4X4_50": cv.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv.aruco.DICT_5X5_1000,
    "DICT_6X6_50": cv.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv.aruco.DICT_6X6_250,
    "DICT_6X6_1000": cv.aruco.DICT_6X6_1000,
    "DICT_7X7_50": cv.aruco.DICT_7X7_50,
    "DICT_7X7_100": cv.aruco.DICT_7X7_100,
    "DICT_7X7_250": cv.aruco.DICT_7X7_250,
    "DICT_7X7_1000": cv.aruco.DICT_7X7_1000,
    "DICT_ARUCO_ORIGINAL": cv.aruco.DICT_ARUCO_ORIGINAL,
    "DICT_APRILTAG_16h5": cv.aruco.DICT_APRILTAG_16h5,
    "DICT_APRILTAG_25h9": cv.aruco.DICT_APRILTAG_25h9,
    "DICT_APRILTAG_36h10": cv.aruco.DICT_APRILTAG_36h10,
    "DICT_APRILTAG_36h11": cv.aruco.DICT_APRILTAG_36h11
}
# load the input image from disk and resize it
print("[INFO] loading image...")
image = cv.imread(args["image"])
#image = imutils.resize(image, width=1600)
# verify that the supplied ArUCo tag exists and is supported by
# OpenCV
if ARUCO_DICT.get(args["type"], None) is None:
    print("[INFO] ArUCo tag of '{}' is not supported".format(
        args["type"]))
    sys.exit(0)
# load the ArUCo dictionary, grab the ArUCo parameters, and detect
# the markers
print("[INFO] detecting '{}' tags...".format(args["type"]))
arucoDict = cv.aruco.getPredefinedDictionary(ARUCO_DICT[args["type"]])
arucoParams = cv.aruco.DetectorParameters()
detector = cv.aruco.ArucoDetector(arucoDict, arucoParams)
frame = cv.imread(args["image"])
corners, ids, rejected = detector.detectMarkers(frame)
# verify *at least* one ArUco marker was detected
if len(corners) > 0:
    # flatten the ArUco IDs list
    ids = ids.flatten()
    # loop over the detected ArUCo corners
    for (markerCorner, markerID) in zip(corners, ids):
        # extract the marker corners (which are always returned in
        # top-left, top-right, bottom-right, and bottom-left order)
        corners = markerCorner.reshape((4, 2))
        (topLeft, topRight, bottomRight, bottomLeft) = corners
        # convert each of the (x, y)-coordinate pairs to integers
        topRight = (int(topRight[0]), int(topRight[1]))
        bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
        bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
        topLeft = (int(topLeft[0]), int(topLeft[1]))
        # draw the bounding box of the ArUCo detection
        cv.line(image, topLeft, topRight, (0, 255, 0), 2)
        cv.line(image, topRight, bottomRight, (0, 255, 0), 2)
        cv.line(image, bottomRight, bottomLeft, (0, 255, 0), 2)
        cv.line(image, bottomLeft, topLeft, (0, 255, 0), 2)
        # compute and draw the center (x, y)-coordinates of the ArUco
        # marker
        cX = int((topLeft[0] + bottomRight[0]) / 2.0)
        cY = int((topLeft[1] + bottomRight[1]) / 2.0)
        cv.circle(image, (cX, cY), 4, (0, 0, 255), -1)
        # draw the ArUco marker ID on the image
        cv.putText(image, str(markerID),
                    (topLeft[0], topLeft[1] - 15), cv.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 2)
        print("[INFO] ArUco marker ID: {}".format(markerID))
        # show the output image
        cv.imshow("Image", image)
    #press any key to end
    cv.waitKey(0)
#end aruco

#code reference: https://pyimagesearch.com/2020/12/21/detecting-aruco-markers-with-opencv-and-python/
#
#

#if you use early opencv versions, please note:
#https://stackoverflow.com/questions/74964527/attributeerror-module-cv2-aruco-has-no-attribute-dictionary-get
    #API changed for 4.7.x, use the following functions:

    #import cv2 as cv

    #dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_250)
    #parameters =  cv.aruco.DetectorParameters()
    #detector = cv.aruco.ArucoDetector(dictionary, parameters)

    #frame = cv.imread(...)

    #markerCorners, markerIds, rejectedCandidates = detector.detectMarkers(frame)