import math
import numpy as np


def goal_distance(goal_a, goal_b):
	return np.linalg.norm(goal_a - goal_b, ord=2)


def goal_distance_obs(obs):
	return goal_distance(obs['achieved_goal'], obs['desired_goal'])


def quat2eul(array):
	# from "Energy-Based Hindsight Experience Prioritization"
	w = array[0]
	x = array[1]
	y = array[2]
	z = array[3]
	ysqr = y * y
	t0 = +2.0 * (w * x + y * z)
	t1 = +1.0 - 2.0 * (x * x + ysqr)
	X = math.atan2(t0, t1)
	t2 = +2.0 * (w * y - z * x)
	t2 = +1.0 if t2 > +1.0 else t2
	t2 = -1.0 if t2 < -1.0 else t2
	Y = math.asin(t2)
	t3 = +2.0 * (w * z + x * y)
	t4 = +1.0 - 2.0 * (ysqr + z * z)
	Z = math.atan2(t3, t4)
	result = np.array([X, Y, Z])
	return result


def eul2quat(array):
	X = array[0]
	Y = array[1]
	Z = array[2]

	w = np.cos(X / 2) * np.cos(Y / 2) * np.cos(Z / 2) + np.sin(X / 2) * np.sin(Y / 2) * np.sin(Z / 2)
	x = np.sin(X / 2) * np.cos(Y / 2) * np.cos(Z / 2) - np.cos(X / 2) * np.sin(Y / 2) * np.sin(Z / 2)
	y = np.cos(X / 2) * np.sin(Y / 2) * np.cos(Z / 2) + np.sin(X / 2) * np.cos(Y / 2) * np.sin(Z / 2)
	z = np.cos(X / 2) * np.cos(Y / 2) * np.sin(Z / 2) - np.sin(X / 2) * np.sin(Y / 2) * np.cos(Z / 2)

	result = np.array([w, x, y, z])
	return result
