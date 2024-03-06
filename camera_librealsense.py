import time
import os
import imutils
import cv2 as cv
import numpy as np
import pyrealsense2 as rs

from matplotlib import pyplot as plt

# Color boundaries
RED = [(50, 0, 0), (150, 30, 30)]
BLUE = [(90, 120, 120), (100, 255, 255)]
GREEN2 = [(80, 150, 120), (100, 180, 150)]
GREEN = [(50, 60, 130), (70, 190, 240)]
ORANGE = [(8, 120, 120), (25, 255, 255)]

# Transformation Matrix to original world frame in Mujoco with realsense intrinsics
R_rs = np.array([
    [0, 0, -1],
    [1, 0, 0],
    [0, -1, 0],
])
T_rs = np.array([2.41, 0.78, 0.57]).reshape(3, 1)
# Transformation Matrix to original world frame in mujoco with opencv intrinsics
R_cv = np.array([
    [0, 0, -1],
    [1, 0, 0],
    [0, -1, 0],
])
T_cv = (np.array([1.51, 0.05, 0.12]) + np.array([0.8, 0.75, 0.4])).reshape(3, 1)

# base 0.8, 0.75, 0.4

class Camera:
    def __init__(self):
        # for librealsense
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        self.config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)  # 1080p resolution
        self.config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)

        self.arucoDict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_APRILTAG_36h11)
        self.arucoParams = cv.aruco.DetectorParameters()
        self.detector = cv.aruco.ArucoDetector(self.arucoDict, self.arucoParams)

        # alignment
        align_to = rs.stream.color
        self.align = rs.align(align_to)

        # Camera intrinsics from calibration
        try:
            self.mtx = np.load("./calibration/mtx.npy")
            self.dist = np.load("./calibration/dist.npy")
        except Exception as e:
            pass

    def start(self):
        self.pipeline.start(self.config)
        time.sleep(1)

    def get_frames(self):
        frames = self.pipeline.wait_for_frames()

        # Align the depth frame to color frame
        aligned_frames = self.align.process(frames)
        depth_frame = np.asanyarray(aligned_frames.get_depth_frame().get_data())
        color_frame = np.asanyarray(frames.get_color_frame().get_data())

        return color_frame, depth_frame

    def get_frames_2(self):

        frames = self.pipeline.wait_for_frames()  # 等待获取图像帧，获取颜色和深度的框架集
        aligned_frames = self.align.process(frames)  # 获取对齐帧，将深度框与颜色框对齐

        aligned_depth_frame = aligned_frames.get_depth_frame()  # 获取对齐帧中的的depth帧
        aligned_color_frame = aligned_frames.get_color_frame()  # 获取对齐帧中的的color帧

        #### 获取相机参数 ####
        depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics  # 获取深度参数（像素坐标系转相机坐标系会用到）
        color_intrin = aligned_color_frame.profile.as_video_stream_profile().intrinsics  # 获取相机内参

        #### 将images转为numpy arrays ####
        color_frame = np.asanyarray(aligned_color_frame.get_data())  # BGR图
        depth_frame = np.asanyarray(aligned_depth_frame.get_data())  # 深度图（默认16位）

        return color_intrin, depth_intrin, color_frame, depth_frame, aligned_depth_frame

    def stop(self):
        self.pipeline.stop()

    def plot_frames(self):
        # # Sometimes bodies might not be recognized from the image frames, so we try until we got everything
        # while True:
        #     color_frame, depth_frame = self.get_frames()
        #
        #     obj_pixel = self.detect_by_color(color_frame, depth_frame, RED)
        #     r_mask = cv.inRange(color_frame, *RED)
        #     goal_pixel = self.detect_by_color(color_frame, depth_frame, BLUE)
        #     b_mask = cv.inRange(color_frame, *BLUE)
        #     obst_pixel = self.detect_aruco_marker(color_frame, depth_frame)
        #
        #     # If all detections are successful, break out of the loop
        #     if all([obj_pixel, goal_pixel, obst_pixel]):
        #         t = int(time.perf_counter())
        #         color_filename = os.path.join("./data", f"color_frame_{t}")
        #         np.save(color_filename, color_frame)
        #         depth_filename = os.path.join("./data", f"depth_frame_{t}")
        #         np.save(depth_filename, depth_frame)
        #         break

        # Load and process each image
        dir = "./data"
        for file in os.listdir(dir):
            if file.startswith("color"):
                color_frame = np.load(os.path.join(dir, file))
                depth_frame = np.load(os.path.join(dir, "depth" + file[5:]))

                obj_pixel = self.detect_by_color(color_frame, depth_frame, GREEN)
                r_mask = cv.inRange(color_frame, *GREEN)
                goal_pixel = self.detect_by_color(color_frame, depth_frame, BLUE)
                b_mask = cv.inRange(color_frame, *BLUE)
                obst_pixel = self.detect_aruco_marker(color_frame, depth_frame)
            else:
                continue

            fig, ax = plt.subplots(2, 2, figsize=(12, 6))

            ax[0][0].imshow(color_frame)
            ax[0][0].scatter(obj_pixel[0], obj_pixel[1], c='r', s=100, marker='x')
            ax[0][0].scatter(goal_pixel[0], goal_pixel[1], c='b', s=100, marker='x')
            ax[0][0].scatter(obst_pixel[0], obst_pixel[1], c='g', s=100, marker='x')
            ax[0][0].set_title("Color Frame")

            ax[1][0].imshow(depth_frame, cmap='gray')
            ax[1][0].scatter(obj_pixel[0], obj_pixel[1], c='r', s=100, marker='x')
            ax[1][0].scatter(goal_pixel[0], goal_pixel[1], c='b', s=100, marker='x')
            ax[1][0].scatter(obst_pixel[0], obst_pixel[1], c='g', s=100, marker='x')
            ax[1][0].set_title("Depth Frame")

            ax[0][1].imshow(r_mask)
            ax[0][1].scatter(obj_pixel[0], obj_pixel[1], c='r', s=100, marker='x')
            ax[0][1].set_title("Red Mask")

            ax[1][1].imshow(b_mask)
            ax[1][1].scatter(goal_pixel[0], goal_pixel[1], c='b', s=100, marker='x')
            ax[1][1].set_title("Blue Mask")

            plt.show()

    def detect_by_color(self, color_frame, depth_frame, color_bounds):
        # get mask for red object
        hsv = cv.cvtColor(color_frame, cv.COLOR_BGR2HSV)
        mask = cv.inRange(hsv, color_bounds[0], color_bounds[1])
        # Find contours
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        # assuming the largest contour
        if contours:
            largest = max(contours, key=cv.contourArea)
            x, y, w, h = cv.boundingRect(largest)
            center_x = x + w // 2
            center_y = y + h // 2
            depth = depth_frame[center_y, center_x]
            return center_x, center_y, depth
        return None

    def detect_aruco_marker(self, color_frame, depth_frame):
        # detect Aruco markers
        #corners, ids, _ = cv.aruco.detectMarkers(color_frame, self.arucoDict, parameters=self.arucoParams)
        detector = cv.aruco.ArucoDetector(self.arucoDict, self.arucoParams)
        (corners, ids, rejected) = detector.detectMarkers(color_frame)

        if ids is not None:
            top_left = tuple(corners[0][0][0])
            bottom_right = tuple(corners[0][0][2])

            center_x = (top_left[0] + bottom_right[0]) // 2
            center_y = (top_left[1] + bottom_right[1]) // 2
            depth = depth_frame[int(center_y), int(center_x)]
            return center_x, center_y, depth
        return None

    def pixel_to_world_cv2(self, pixel):

        x, y, depth = pixel
        depth = depth / 1000  # conver to m

        points = np.array([[x, y]], dtype=np.float32).reshape(-1, 1, 2)

        # Undistort the pixel
        undistorted_points = cv.undistortPoints(points, self.mtx, self.dist, P=self.mtx)

        # Extract focal lengths and principal point from the camera matrix
        fx, fy = self.mtx[0, 0], self.mtx[1, 1]
        cx, cy = self.mtx[0, 2], self.mtx[1, 2]

        # Compute 3D coordinates
        u, v = undistorted_points[0, 0]
        x = ((u - cx) / fx) * depth
        y = ((v - cy) / fy) * depth
        z = depth

        # # Transform to World system
        point = np.array([x, y, z]).reshape(3, 1)
        point = np.dot(R_cv, point) + T_cv
        return point

    def get_projection_params(self):
        profile = self.pipeline.get_active_profile()
        color_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
        depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))

        intrinsics = color_profile.get_intrinsics()
        extrinsics = color_profile.get_extrinsics_to(depth_profile)
        return intrinsics, extrinsics

    def pixel_to_world(self, pixel, intrinsics, extrinsics):
        z = pixel[2]/1000
        x = ((pixel[0] - intrinsics.ppx) / intrinsics.fx) * z
        y = ((pixel[1] - intrinsics.ppy) / intrinsics.fy) * z

        R = np.array(extrinsics.rotation).reshape(3, 3)
        T = np.array(extrinsics.translation).reshape(3, 1)
        point = np.array([x, y, z]).reshape(3, 1)
        # Transform to Camera system
        point = np.dot(R, point) + T
        # # Transform to World system
        point = np.dot(R_rs, point) + T_rs
        return point

    def get_body_pos_rs(self):
        # Sometimes bodies might not be recognized from the image frames, so we try until we got everything
        while True:
            color_frame, depth_frame = self.get_frames()
            intrinsics, extrinsics = self.get_projection_params()

            obj_pixel = self.detect_by_color(color_frame, depth_frame, RED)
            goal_pixel = self.detect_by_color(color_frame, depth_frame, BLUE)
            obst_pixel = self.detect_aruco_marker(color_frame, depth_frame)

            # If all detections are successful, break out of the loop
            if all([obj_pixel, goal_pixel, obst_pixel]):
                break

        body_pos = {
            'object': self.pixel_to_world(obj_pixel, intrinsics, extrinsics).flatten(),
            'goal': self.pixel_to_world(goal_pixel, intrinsics, extrinsics).flatten(),
            'obst': self.pixel_to_world(obst_pixel, intrinsics, extrinsics).flatten()
        }
        return body_pos

    def get_body_pos_cv(self):
        # Sometimes bodies might not be recognized from the image frames, so we try until we got everything
        while True:
            color_frame, depth_frame = self.get_frames()

            obj_pixel = self.detect_by_color(color_frame, depth_frame, GREEN)
            goal_pixel = self.detect_by_color(color_frame, depth_frame, BLUE)
            # obst_pixel = self.detect_aruco_marker(color_frame, depth_frame)
            obst1_pixel = self.detect_by_color(color_frame, depth_frame, BLUE)
            obst2_pixel = self.detect_by_color(color_frame, depth_frame, ORANGE)

            # cv.imshow("Frame", color_frame)
            # key = cv.waitKey(1) & 0xFF
            # # if the `q` key was pressed, break from the loop
            # if key == ord("q"):
            #     break
            # If all detections are successful, break out of the loop
            if all([obj_pixel, goal_pixel, obst1_pixel, obst2_pixel]):
                break

        body_pos = {
            'object': self.pixel_to_world_cv2(obj_pixel).flatten(),
            'goal': self.pixel_to_world_cv2(goal_pixel).flatten(),
            'obst1': self.pixel_to_world_cv2(obst1_pixel).flatten(),
            'obst2': self.pixel_to_world_cv2(obst2_pixel).flatten()
        }

        return body_pos


