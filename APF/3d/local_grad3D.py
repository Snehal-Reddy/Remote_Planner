# In order to launch execute:
# python3 gradient_interactive.py

import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from matplotlib import collections
from scipy.ndimage.morphology import distance_transform_edt as bwdist
from math import *
import random
import time

from progress.bar import FillingCirclesBar
from task import *
from threading import Thread
from multiprocessing import Process
import os
from mpl_toolkits import mplot3d

def meters2grid(pose_m, nrows=50, ncols=50, nd = 50):
    # [0, 0](m) -> [250, 250]
    # [1, 0](m) -> [250+100, 250]
    # [0,-1](m) -> [250, 250-100]
    pose_on_grid = np.array(pose_m)*2 + np.array([ncols/2, nrows/2, nd/2])
    return np.array( pose_on_grid, dtype=int)
# def grid2meters(pose_grid, nrows=50, ncols=50, nd=50):
#     # [250, 250] -> [0, 0](m)
#     # [250+100, 250] -> [1, 0](m)
#     # [250, 250-100] -> [0,-1](m)
#     pose_meters = ( np.array(pose_grid) - np.array([ncols/2, nrows/2, nd/2]) ) / 100.0
#     return pose_meters

def gradient_planner(f, current_point, ncols=50, nrows=50, nd = 50, movement_rate=0.06):
    """
    GradientBasedPlanner : This function computes the next_point
    given current location, goal location and potential map, f.
    It also returns mean velocity, V, of the gradient map in current point.
    """
    [gy, gx, gz] = np.gradient(-f);
    iy, ix, iz = np.array( meters2grid(current_point), dtype=int )
    # gz = 0
    # iz = 0  
    w = 20 # smoothing window size for gradient-velocity
    vx = np.mean(gx[ix-int(w/2) : ix+int(w/2), iy-int(w/2) : iy+int(w/2), iz-int(w/2) : iz+int(w/2)])
    vy = np.mean(gy[ix-int(w/2) : ix+int(w/2), iy-int(w/2) : iy+int(w/2), iz-int(w/2) : iz+int(w/2)])
    vz = np.mean(gz[ix-int(w/2) : ix+int(w/2), iy-int(w/2) : iy+int(w/2), iz-int(w/2) : iz+int(w/2)])
    # vz = 0
    V = np.array([vx, vy, vz])
    dt = 0.2 / norm(V);
    next_point = current_point + dt*V;

    return next_point, V

def combined_potential(obstacles_poses, goal, nrows=50, ncols=50, nd=50):
    global num_random_obstacles
    obstacles_map = map(obstacles_poses)
    goal = meters2grid(goal)
    d = bwdist(obstacles_map==0); #bwdist 3d applicable?
    d2 = (d/2.) + 1; # Rescale and transform distances
    d0 = 2;
    nu = 300;
    repulsive = nu*((1./d2 - 1./d0)**2);
    repulsive [d2 > d0] = 0;

    [x, y, z] = np.meshgrid(np.arange(ncols), np.arange(nrows), np.arange(nd))
    xi = 1/50.;
    attractive = xi*(num_random_obstacles/20) * ( (x - goal[0])**2 + (y - goal[1])**2 + (z - goal[2])**2);
    """ Combine terms """
    f = attractive + repulsive;
    return f

def map(obstacles_poses, nrows=50, ncols=50, nd=50):
    obstacles_map = np.zeros((nrows, ncols, nd));
    [x, y, z] = np.meshgrid(np.arange(ncols), np.arange(nrows), np.arange(nd))
    for pose in obstacles_poses:
        pose = meters2grid(pose)
        x0 = pose[0]; y0 = pose[1]; z0 = pose[2]
        # cylindrical obstacles
        t = ((x - x0)**2 + (y - y0)**2 + (z - z0)**2) < (10*R_obstacles)**2
        obstacles_map[t] = 1;
    # rectangular obstacles
    return obstacles_map

def move_obstacles(obstacles_poses):
    global obstacles_goal_poses
    for p in range(len(obstacles_poses)):
        pose = obstacles_poses[p]
        goal = obstacles_goal_poses[p]
        dx, dy, dz = (goal - pose) / norm(goal-pose) * 0.1#random.uniform(0,0.05)
        pose[0] += dx; pose[1] += dy; pose[2] += dy;
        if(dx**2 + dy**2 + dz**2 < 1):
            obstacles_goal_poses = np.random.uniform(low=-5, high=5, size=(num_random_obstacles,3))
    return obstacles_poses



""" initialization """
animate              = 1   
num_random_obstacles = 20  
num_robots           = 1   
moving_obstacles     = 1   
draw_gradients       = 0   

max_its              = 500 
progress_bar = FillingCirclesBar('Number of Iterations', max=max_its)
should_write_movie = 0; movie_file_name = os.getcwd()+'/videos/output.avi'
movie_writer = get_movie_writer(should_write_movie, 'Simulation Potential Fields', movie_fps=10., plot_pause_len=0.01)

R_obstacles = 0.05 # [m]
R_swarm     = 0.3 # [m]
start = np.array([-5, -5,-5]); goal = np.array([2.5,4,3])
V0 = (goal - start) / norm(goal-start)    # initial movement direction, |V0| = 1
U0 = np.array([-V0[1], V0[0]]) / norm(V0) # perpendicular to initial movement direction, |U0|=1
imp_pose_prev = np.array([0, 0, 0])
imp_vel_prev  = np.array([0, 0, 0])
imp_time_prev = time.time()

obstacles_poses      = np.random.uniform(low=-5, high=5, size=(num_random_obstacles,3)) # randomly located obstacles
obstacles_goal_poses = np.random.uniform(low=-5, high=5, size=(num_random_obstacles,3)) # randomly located obstacles goal poses


route1 = start # leader
current_point1 = start
robots_poses = start
routes = route1
# centroid_route = [ sum([p[0] for p in robots_poses])/len(robots_poses), sum([p[1] for p in robots_poses])/len(robots_poses),  sum([p[2] for p in robots_poses])/len(robots_poses) ]
des_poses = robots_poses
vels = []
norm_vels = []

start_time = time.time()


for i in range(max_its):
    if moving_obstacles: 
    	obstacles_poses = move_obstacles(obstacles_poses)

    visible_obs = []
    for obs in obstacles_poses:
        if(np.linalg.norm(current_point1-obs)<5):
            visible_obs.append(obs)
    print(len(visible_obs), len(obstacles_poses))
    f1 = combined_potential(visible_obs, goal)
    # print(f1)
    des_poses, vels = gradient_planner(f1, current_point1)
    direction = ( goal - des_poses ) / norm(goal - des_poses)
    norm_vels.append(norm(vels))

    v = direction
    u = np.array([-v[1], v[0]])

    routes = np.vstack([routes, des_poses])

    current_point1 = des_poses # update current point of the leader
    print(current_point1)

    for obs in obstacles_poses:
        if norm(current_point1-obs)<0.1:
            print("Booom !")

    dist_to_goal = norm(current_point1 - goal)
    if dist_to_goal < 0.2:
        print('\nReached the goal')
        break

    # plt.cla()

    draw_map(start, goal, obstacles_poses, current_point1, routes, vels)

    if animate:
        plt.draw()
        plt.pause(0.01)

    # print('Current simulation time: ', time.time()-start_time)
print('\nDone')
progress_bar.finish()
end_time = time.time()
print('Simulation execution time: ', round(end_time-start_time,2))
plt.show()
