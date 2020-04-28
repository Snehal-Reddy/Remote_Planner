import contextlib
from matplotlib import animation as anim
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from math import *
import time
from mpl_toolkits import mplot3d


def draw_map(start, goal, obstacles_poses, current_point1, routes=None, vel1=None):
	ax = plt.axes(projection='3d')
	ax.scatter3D(start[0], start[1], start[2], color='yellow', s=10);
	ax.scatter3D(goal[0], goal[1], goal[2], color='green', s=10);
	# plt.xlabel('X')
	# plt.ylabel('Y')
	# ax = plt.gca()
	u = np.linspace(0, np.pi, 10)
	v = np.linspace(0, 2 * np.pi, 10)

	x = current_point1[0]+4*np.outer(np.sin(u), np.sin(v))
	y = current_point1[1]+4*np.outer(np.sin(u), np.cos(v))
	z = current_point1[2]+4*np.outer(np.cos(u), np.ones_like(v))
	ax.plot_wireframe(x, y, z, color="g")

	for pose in obstacles_poses:
		ax.scatter3D(pose[0], pose[1], pose[2], color='red', s=50);

	if vel1 is not None: 
		ax.quiver(
		current_point1[0], current_point1[1], current_point1[2], # <-- starting point of vector
		vel1[0]*10, vel1[2]*10, vel1[2]*10,  # <-- directions of vector
		color = 'red', alpha = .8, lw = 3,
		)	

	# ax.scatter3D([r[0] for r in routes], [r[0] for r in routes], [r[0] for r in routes], color = 'blue', s = 10)

	ax.scatter3D(current_point1[0], current_point1[1], current_point1[2], color='blue', s = 50)

	
# def draw_robots(current_point1, routes=None, vel1=None):
# 	# pass
# 	# ax = plt.axes(projection='3d')
# 	if vel1 is not None: 
# 		ax.quiver(
# 		current_point1[0], current_point1[1], current_point1[2], # <-- starting point of vector
# 		vel1[0], vel1[2], vel1[2],  # <-- directions of vector
# 		color = 'red', alpha = .8, lw = 3,
# 		)	

# 	ax.scatter3D([r[0] for r in routes], [r[0] for r in routes], [r[0] for r in routes], color = 'green', s = 10)

# 	ax.scatter3D(current_point1[0], current_point1[1], current_point1[2], color='blue', s = 20)
 

def get_movie_writer(should_write_movie, title, movie_fps, plot_pause_len):
	"""
	:param should_write_movie: Indicates whether the animation of SLAM should be written to a movie file.
	:param title: The title of the movie with which the movie writer will be initialized.
	:param movie_fps: The frame rate of the movie to write.
	:param plot_pause_len: The pause durations between the frames when showing the plots.
	:return: A movie writer that enables writing MP4 movie with the animation from SLAM.
	"""

	get_ff_mpeg_writer = anim.writers['ffmpeg']
	metadata = dict(title=title, artist='matplotlib', comment='Potential Fields Formation Navigation')
	movie_fps = min(movie_fps, float(1. / plot_pause_len))

	return get_ff_mpeg_writer(fps=movie_fps, metadata=metadata)

@contextlib.contextmanager
def get_dummy_context_mgr():
	"""
	:return: A dummy context manager for conditionally writing to a movie file.
	"""
	yield None


# HUMAN VELOCITY CALCULATION
hum_time_array = np.ones(10)
hum_pose_array = np.array([ np.ones(10), np.ones(10), np.ones(10) ])
def hum_vel(human_pose):

	for i in range(len(hum_time_array)-1):
		hum_time_array[i] = hum_time_array[i+1]
	hum_time_array[-1] = time.time()

	for i in range(len(hum_pose_array[0])-1):
		hum_pose_array[0][i] = hum_pose_array[0][i+1]
		hum_pose_array[1][i] = hum_pose_array[1][i+1]
		hum_pose_array[2][i] = hum_pose_array[2][i+1]
	hum_pose_array[0][-1] = human_pose[0]
	hum_pose_array[1][-1] = human_pose[1]
	hum_pose_array[2][-1] = human_pose[2]

	vel_x = (hum_pose_array[0][-1]-hum_pose_array[0][0])/(hum_time_array[-1]-hum_time_array[0])
	vel_y = (hum_pose_array[1][-1]-hum_pose_array[1][0])/(hum_time_array[-1]-hum_time_array[0])
	vel_z = (hum_pose_array[2][-1]-hum_pose_array[2][0])/(hum_time_array[-1]-hum_time_array[0])

	hum_vel = np.array( [vel_x, vel_y, vel_z] )

	return hum_vel


def euler_from_quaternion(q):
	"""
	Intrinsic Tait-Bryan rotation of xyz-order.
	"""
	q = q / np.linalg.norm(q)
	qx, qy, qz, qw = q
	roll = atan2(2.0*(qy*qz + qw*qx), qw*qw - qx*qx - qy*qy + qz*qz)
	pitch = asin(-2.0*(qx*qz - qw*qy))
	yaw = atan2(2.0*(qx*qy + qw*qz), qw*qw + qx*qx - qy*qy - qz*qz)
	return roll, pitch, yaw