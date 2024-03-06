import cv2
import pyrealsense2 as rs
import numpy as np
from numpy import sin, cos
from camera_librealsense import Camera
import math

############################################

# w h分别是棋盘格模板长边和短边规格（角点个数）
w = 7
h = 6
board = 25

cam = Camera()
cam.start()

try:
	while True:
		# 采集图像
		img, _ = cam.get_frames()  # 打开第0通道数据流

		# 找棋盘格角点
		# 阈值
		criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
		# print(cv2.TERM_CRITERIA_EPS,'',cv2.TERM_CRITERIA_MAX_ITER)

		# 世界坐标系中的棋盘格点,例如(0,0,0), (1,0,0), (2,0,0) ....,(8,5,0)，去掉Z坐标，记为二维矩阵，认为在棋盘格这个平面上Z=0
		objp = np.zeros((w * h, 3), np.float32)  # 构造0矩阵，88行3列，用于存放角点的世界坐标
		objp[:, :2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2)  # 三维网格坐标划分

		# 储存棋盘格角点的世界坐标和图像坐标对
		objpoints = []  # 在世界坐标系中的三维点
		imgpoints = []  # 在图像平面的二维点

		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		# 粗略找到棋盘格角点 这里找到的是这张图片中角点的亚像素点位置，共11×8 = 88个点，gray必须是8位灰度或者彩色图，（w,h）为角点规模
		ret, corners = cv2.findChessboardCorners(gray, (w, h))

		# 如果找到足够点对，将其存储起来
		if ret:
			# 精确找到角点坐标
			corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

			worldpoint = objp * board  # 棋盘格的宽度为
			imagepoint = np.squeeze(corners)  # 将corners降为二维

			(success, rvec, tvec) = cv2.solvePnP(worldpoint, imagepoint, cam.mtx, cam.dist)

			distance = math.sqrt(tvec[0] ** 2 + tvec[1] ** 2 + tvec[2] ** 2) / 10  # 测算距离
			rvec_matrix = cv2.Rodrigues(rvec)[0]
			proj_matrix = np.hstack((rvec_matrix, rvec))
			eulerAngles = -cv2.decomposeProjectionMatrix(proj_matrix)[6]  # 欧拉角
			rot_matrix = cv2.decomposeProjectionMatrix(proj_matrix)[1]  #
			roll, pitch, yaw = eulerAngles[0], eulerAngles[1], eulerAngles[2]
			# 显示参数文字
			cv2.putText(img, "%.2fcm,%.2f,%.2f,%.2f" % (distance, roll, pitch, yaw),
						(100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
			# 将角点在图像上显示
			cv2.drawChessboardCorners(img, (w, h), corners, ret)
			cv2.imshow('findCorners', img)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
finally:
	cam.stop()