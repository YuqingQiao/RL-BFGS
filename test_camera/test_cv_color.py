import cv2
import pyrealsense2 as rs
import numpy as np
from numpy import sin, cos
from camera_librealsense import Camera


############################################
################## TBD #####################
px = 0.420
py = -0.700
pz = 0.260
phi_x = -90 * np.pi / 180
theta_y = 0 * np.pi / 180
eta_z = 0 * np.pi / 180
############################################
############################################

T_ee_cam = np.array([
	[cos(eta_z) * cos(theta_y), cos(eta_z) * sin(theta_y) * sin(phi_x) - sin(eta_z) * cos(phi_x),
	 cos(eta_z) * sin(theta_y) * cos(phi_x) + sin(eta_z) * sin(phi_x), px],
	[sin(eta_z) * cos(theta_y), sin(eta_z) * sin(theta_y) * sin(phi_x) + cos(eta_z) * cos(phi_x),
	 sin(eta_z) * sin(theta_y) * cos(phi_x) - cos(eta_z) * sin(phi_x), py],
	[-sin(theta_y), cos(theta_y) * sin(phi_x), cos(theta_y) * cos(phi_x), pz],
	[0.0, 0.0, 0.0, 1.0]
])

''' 
获取对齐图像帧与相机参数
'''
cam = Camera()
cam.start()

''' 
获取随机点三维坐标
'''


def get_3d_camera_coordinate(depth_pixel, aligned_depth_frame, depth_intrin):
	x = depth_pixel[0]
	y = depth_pixel[1]
	dis = aligned_depth_frame.get_distance(x, y)  # 获取该像素点对应的深度
	camera_coordinate = rs.rs2_deproject_pixel_to_point(depth_intrin, depth_pixel, dis)
	return dis, camera_coordinate  # return type list


def get_base_coordinate(camera_coordinate: list):
	'''
    given point in camera frame, calculate the pose in robot base frame.
    '''
	camera_coordinate.append(1.0)
	camera_coordinate = np.array(camera_coordinate)
	return np.matmul(T_ee_cam, camera_coordinate)[:3]


'''
marker
'''


def set_marker(color_frame, mask_center, camera_coordinate, dis, color=[0, 0, 255], st=''):
	cv2.circle(color_frame, mask_center, 5, color, thickness=-1)
	cv2.putText(color_frame, f"{st} Dis:{dis:.4f} m", (mask_center[0] + 5, mask_center[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
				color, )
	cv2.putText(color_frame, f"{st} Cam X:{camera_coordinate[0]:.4f} m", (mask_center[0] + 5, mask_center[1] + 20),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, color)
	cv2.putText(color_frame, f"{st} Cam Y:{camera_coordinate[1]:.4f} m", (mask_center[0] + 5, mask_center[1] + 40),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, color)
	cv2.putText(color_frame, f"{st} Cam Z:{camera_coordinate[2]:.4f} m", (mask_center[0] + 5, mask_center[1] + 60),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, color)
	print(f'cam x: {camera_coordinate[0]:.4f}, y: {camera_coordinate[1]:.4f}, z: {camera_coordinate[2]:.4f}')

	###########################################################################
	# for test
	ee_pos = get_base_coordinate(camera_coordinate)
	print(f'base x: {ee_pos[0]:.4f}, y: {ee_pos[1]:.4f}, z: {ee_pos[2]:.4f}')
	###########################################################################

	print('------------------')

	return color_frame


'''
ColorFilter
'''


class ColorFilter():
	def __init__(self) -> None:
		# color filtering range
		# [((H_lower, S_lower, V_lower),(H_upper, S_upper, V_upper)),..]
		self.colorRange = []

	def __call__(self, img):
		# final generated masking
		finalMask = np.zeros_like(img)[:, :, 0]

		# convert BGR frame into HSV format
		hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

		# read each HSV range limits
		for each in self.colorRange:
			lower, upper = each

			# create masking using each range
			mask = cv2.inRange(hsv, lower, upper)  # shape (480, 640)

			mask_yx_coord = np.column_stack(np.where(mask == 255))
			if mask_yx_coord.shape[0] == 0:
				center_x = 320
				center_y = 240
			else:
				center_y, center_x = np.mean(mask_yx_coord, axis=0)

			mask_center = [int(center_x), int(center_y)]  # Warning: if none then [nan, nan]

			# merge the masking, target color = color 1 + color 2 ...
			finalMask = cv2.bitwise_or(finalMask, mask)

		# set up the final merged masked image
		merged_img = self.setMask(img, finalMask)

		return merged_img, mask, mask_center

	def setMask(self, img, mask):
		# split the channel
		channels = cv2.split(img)
		result = []
		for i in range(len(channels)):
			result.append(cv2.bitwise_and(channels[i], mask))
		# append masking for each channel
		masked_img = cv2.merge(result)

		return masked_img


item_color_filter = ColorFilter()
goal_color_filter = ColorFilter()
red_colorRange = [
	((0, 120, 120), (5, 255, 255)),  # red color range 1 HSV
	((175, 120, 120), (180, 255, 255))  # red color range 2 HSV
	# you can add as many colors as you would like
	# final detected merged color == color 1 + color 2 + ..
]
green_colorRange = [
	((50, 60, 130), (70, 190, 240)),  # green color range 1 HSV
]
# (35, 43, 46), (77, 255, 255))
# (35, 43, 146), (77, 155, 255)
# (80, 150, 120), (100, 180, 150)
# goal_color_filter
blue_colorRange = [
	((90, 120, 120), (100, 255, 255)),
]
orange_colorRange = [
	((8, 120, 120), (25, 255, 255)),
]
item_color_filter.colorRange = blue_colorRange
goal_color_filter.colorRange = blue_colorRange

try:
	while True:

		''' 
        获取对齐图像帧与相机参数
        '''
		color_intrin, depth_intrin, color_frame, depth_frame, aligned_depth_frame = cam.get_frames_2()    # 获取对齐图像与相机参数
		'''
        get object identifying results
        '''
		img_goal_masked, goal_mask, goal_mask_center = goal_color_filter(color_frame)

		goal_dis, goal_camera_coordinate = get_3d_camera_coordinate(
			goal_mask_center, aligned_depth_frame, depth_intrin)

		''' 
        显示图像与标注
        '''
		color_frame = set_marker(color_frame, goal_mask_center, goal_camera_coordinate,
							   goal_dis, color=[0, 255, 0], st='goal')  # green

		#### 显示画面 ####
		cv2.namedWindow("RealSense", cv2.WINDOW_AUTOSIZE)  # set up the window to display the results
		cv2.imshow('Realsense', color_frame)
		key = cv2.waitKey(1)
		if key & 0xFF == ord('q') or key == 27:  # press the keyboard 'q' or 'esc' to terminate the thread
			cv2.destroyAllWindows()
			break
finally:
	cam.stop()



