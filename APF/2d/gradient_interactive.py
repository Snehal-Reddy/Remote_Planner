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
from tasks import *
from threading import Thread
from multiprocessing import Process
import os


def meters2grid(pose_m, nrows=500, ncols=500):
    # [0, 0](m) -> [250, 250]
    # [1, 0](m) -> [250+100, 250]
    # [0,-1](m) -> [250, 250-100]
    pose_on_grid = np.array(pose_m)*100 + np.array([ncols/2, nrows/2])
    return np.array( pose_on_grid, dtype=int)
def grid2meters(pose_grid, nrows=500, ncols=500):
    # [250, 250] -> [0, 0](m)
    # [250+100, 250] -> [1, 0](m)
    # [250, 250-100] -> [0,-1](m)
    pose_meters = ( np.array(pose_grid) - np.array([ncols/2, nrows/2]) ) / 100.0
    return pose_meters

def gradient_planner(f, current_point, ncols=500, nrows=500, movement_rate=0.06):

    [gy, gx] = np.gradient(-f);
    iy, ix = np.array( meters2grid(current_point), dtype=int )
    w = 30 # smoothing window size for gradient-velocity
    vx = np.mean(gx[ix-int(w/2) : ix+int(w/2), iy-int(w/2) : iy+int(w/2)])
    vy = np.mean(gy[ix-int(w/2) : ix+int(w/2), iy-int(w/2) : iy+int(w/2)])
    V = np.array([vx, vy])
    dt = 0.06 / norm(V);
    next_point = current_point + dt*V;

    return next_point, V

def combined_potential(obstacles_poses, goal, nrows=500, ncols=500):
    obstacles_map = map(obstacles_poses)
    goal = meters2grid(goal)
    d = bwdist(obstacles_map==0);
    d2 = (d/100.) + 1; # Rescale and transform distances
    d0 = 2;
    nu = 200;
    repulsive = nu*((1./d2 - 1./d0)**2);
    repulsive [d2 > d0] = 0;
    [x, y] = np.meshgrid(np.arange(ncols), np.arange(nrows))
    xi = 1/700.;
    attractive = xi * ( (x - goal[0])**2 + (y - goal[1])**2 );
    """ Combine terms """
    f = attractive + repulsive;
    return f

def map(obstacles_poses, nrows=500, ncols=500):
    obstacles_map = np.zeros((nrows, ncols));
    [x, y] = np.meshgrid(np.arange(ncols), np.arange(nrows))
    for pose in obstacles_poses:
        pose = meters2grid(pose)
        x0 = pose[0]; y0 = pose[1]
        # cylindrical obstacles
        t = ((x - x0)**2 + (y - y0)**2) < (100*R_obstacles)**2
        obstacles_map[t] = 1;
    # rectangular obstacles
    obstacles_map[400:, 130:150] = 1;
    obstacles_map[130:150, :200] = 1;
    obstacles_map[330:380, 300:] = 1;
    return obstacles_map

def move_obstacles(obstacles_poses, obstacles_goal_poses):
    for p in range(len(obstacles_poses)):
        pose = obstacles_poses[p]; goal = obstacles_goal_poses[p]
        dx, dy = (goal - pose) / norm(goal-pose) * 0.05#random.uniform(0,0.05)
        pose[0] += dx;      pose[1] += dy;

    return obstacles_poses


def formation(num_robots, leader_des, v, R_swarm):
    if num_robots<=1: return []
    u = np.array([-v[1], v[0]])
    des4 = leader_des - v*R_swarm*sqrt(3)                 # follower
    if num_robots==2: return [des4]
    des2 = leader_des - v*R_swarm*sqrt(3)/2 + u*R_swarm/2 # follower
    des3 = leader_des - v*R_swarm*sqrt(3)/2 - u*R_swarm/2 # follower
    if num_robots==3: return [des2, des3]
    
    return [des2, des3, des4]

""" initialization """
animate              = 1   
random_obstacles     = 1   
num_random_obstacles = 8   
num_robots           = 1   
moving_obstacles     = 1   
impedance            = 0   
formation_gradient   = 1   
draw_gradients       = 0   
postprocessing       = 0   

pos_coef             = 3.0    
initialized          = False  
max_its              = 120 
# movie writer
progress_bar = FillingCirclesBar('Number of Iterations', max=max_its)
should_write_movie = 0; movie_file_name = os.getcwd()+'/videos/output.avi'
movie_writer = get_movie_writer(should_write_movie, 'Simulation Potential Fields', movie_fps=10., plot_pause_len=0.01)

R_obstacles = 0.05 # [m]
R_swarm     = 0.3 # [m]
start = np.array([-1.8, 1.8]); goal = np.array([1.8, -1.8])
V0 = (goal - start) / norm(goal-start)    # initial movement direction, |V0| = 1
U0 = np.array([-V0[1], V0[0]]) / norm(V0) # perpendicular to initial movement direction, |U0|=1
imp_pose_prev = np.array([0, 0])
imp_vel_prev  = np.array([0, 0])
imp_time_prev = time.time()

if random_obstacles:
    obstacles_poses      = np.random.uniform(low=-2.5, high=2.5, size=(num_random_obstacles,2)) # randomly located obstacles
    obstacles_goal_poses = np.random.uniform(low=-1.3, high=1.3, size=(num_random_obstacles,2)) # randomly located obstacles goal poses
else:
    obstacles_poses      = np.array([[-2, 1], [1.5, 0.5], [-1.0, 1.5], [0.1, 0.1], [1, -2], [-1.8, -1.8]]) # 2D - coordinates [m]
    obstacles_goal_poses = np.array([[-0, 0], [0.0, 0.0], [ 0.0, 0.0], [0.0, 0.0], [0,  0], [ 0.0,  0.0]])

route1 = start # leader
current_point1 = start
robots_poses = [start] + formation(num_robots, start, V0, R_swarm)
routes = [route1] + robots_poses[1:]
centroid_route = [ sum([p[0] for p in robots_poses])/len(robots_poses), sum([p[1] for p in robots_poses])/len(robots_poses) ]
des_poses = robots_poses
vels = [];
for r in range(num_robots): vels.append([])
norm_vels = [];
for r in range(num_robots): norm_vels.append([])

# variables for postprocessing and performance estimation
area_array = []
start_time = time.time()

fig = plt.figure(figsize=(10, 10))
# with movie_writer.saving(fig, movie_file_name, max_its) if should_write_movie else get_dummy_context_mgr():
for i in range(max_its):
    if moving_obstacles: 
    	obstacles_poses = move_obstacles(obstacles_poses, obstacles_goal_poses)
    f1 = combined_potential(obstacles_poses, goal)
    des_poses[0], vels[0] = gradient_planner(f1, current_point1)
    direction = ( goal - des_poses[0] ) / norm(goal - des_poses[0])
    norm_vels[0].append(norm(vels[0]))

    des_poses[1:] = formation(num_robots, des_poses[0], direction, R_swarm)
    v = direction; u = np.array([-v[1], v[0]])

    for r in range(num_robots):
        routes[r] = np.vstack([routes[r], des_poses[r]])

    current_point1 = des_poses[0] # update current point of the leader

    pp = des_poses
    centroid = [ sum([p[0] for p in pp])/len(pp), sum([p[1] for p in pp])/len(pp) ]
    centroid_route = np.vstack([centroid_route, centroid])
    dist_to_goal = norm(centroid - goal)
    if dist_to_goal < 1.5*R_swarm:
        print('\nReached the goal')
        break

    progress_bar.next()
    plt.cla()

    draw_map(start, goal, obstacles_poses, R_obstacles, f1, draw_gradients=draw_gradients)
    draw_robots(current_point1, routes, num_robots, robots_poses, centroid, vels[0])
    if animate:
        plt.draw()
        plt.pause(0.01)

    if should_write_movie:
        movie_writer.grab_frame()
    # print('Current simulation time: ', time.time()-start_time)
print('\nDone')
progress_bar.finish()
end_time = time.time()
print('Simulation execution time: ', round(end_time-start_time,2))
plt.show()
