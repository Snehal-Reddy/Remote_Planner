import random
import math
import numpy as np
import scipy.spatial
import matplotlib.pyplot as plt
import contextlib
from matplotlib import animation as anim
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from math import *
import time
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D

# parameter
N_SAMPLE = 100  # number of sample_points
N_KNN = 15  # number of edge from one sampled point
MAX_EDGE_LEN = 15.0  # [m] Maximum edge length

show_animation = True

# ax = plt.axes(projection='3d')
fig = plt.figure()
ax  = fig.add_subplot(111, projection = '3d')
class Node:
    """
    Node class for dijkstra search
    """

    def __init__(self, x, y, z, cost, pind):
        self.x = x
        self.y = y
        self.z = z
        self.cost = cost
        self.pind = pind

    def __str__(self):
        return str(self.x) + "," + str(self.y) + "," + str(self.z) + "," + str(self.cost) + "," + str(self.pind)


class KDTree:
    """
    Nearest neighbor search class with KDTree
    """

    def __init__(self, data):
        # store kd-tree
        self.tree = scipy.spatial.cKDTree(data)

    def search(self, inp, k=1):
        """
        Search NN

        inp: input data, single frame or multi frame

        """

        if len(inp.shape) >= 2:  # multi input
            index = []
            dist = []

            for i in inp.T:
                idist, iindex = self.tree.query(i, k=k)
                index.append(iindex)
                dist.append(idist)

            return index, dist

        dist, index = self.tree.query(inp, k=k)
        return index, dist

    def search_in_distance(self, inp, r):
        """
        find points with in a distance r
        """

        index = self.tree.query_ball_point(inp, r)
        return index


def PRM_planning(sx, sy, sz, gx, gy, gz, ox, oy, oz, rr):
    time = 0
    while(1):
        time+=1
        obkdtree = KDTree(np.vstack((ox, oy, oz)).T)
        sample_x, sample_y, sample_z = sample_points(sx, sy, sz, gx, gy, gz, rr, ox, oy, oz, obkdtree)
        if show_animation:
            # ax.scatter3D(sample_x, sample_y, sample_z, color='yellow', s=10);
            ax.scatter(sample_x, sample_y, sample_z, color='y', marker="o")
            # plt.draw()
            # plt.pause(10)
        if(time>2):
            print(len(ox))
        road_map = generate_roadmap(sample_x, sample_y, sample_z, rr, obkdtree)
        print(len(road_map))

        rx, ry, rz = dijkstra_planning(
            sx, sy, sz, gx, gy, gz, ox, oy, oz, rr, road_map, sample_x, sample_y, sample_z)
        if(len(rx)>0):
            break

    return rx, ry, rz


def is_collision(sx, sy, sz, gx, gy, gz, rr, okdtree):
    x = sx
    y = sy
    z = sz
    dx = gx - sx
    dy = gy - sy
    dz = gz - sz
    # yaw = math.atan2(gy - sy, gx - sx)
    theta = math.acos(dz/math.sqrt(dx**2+dy**2+dz**2))
    phi = math.acos(dx/math.sqrt(dx**2+dy**2))
    d = math.sqrt(dx**2+dy**2+dz**2)

    if d >= MAX_EDGE_LEN:
        return True

    D = rr
    nstep = round(d / D)

    for i in range(nstep):
        idxs, dist = okdtree.search(np.array([x, y, z]).reshape(3, 1))
        if dist[0] <= rr:
            return True  # collision
        x += D * math.sin(theta) * math.cos(phi)    
        y += D * math.sin(theta) * math.sin(phi)
        z += D * math.cos(theta)

    # goal point check
    idxs, dist = okdtree.search(np.array([gx, gy, gz]).reshape(3, 1))
    if dist[0] <= rr:
        return True  # collision

    return False  # OK


def generate_roadmap(sample_x, sample_y, sample_z, rr, obkdtree):
    """
    Road map generation

    sample_x: [m] x positions of sampled points
    sample_y: [m] y positions of sampled points
    rr: Robot Radius[m]
    obkdtree: KDTree object of obstacles
    """

    road_map = []
    nsample = len(sample_x)
    skdtree = KDTree(np.vstack((sample_x, sample_y, sample_z)).T)

    for (i, ix, iy, iz) in zip(range(nsample), sample_x, sample_y, sample_z):

        index, dists = skdtree.search(
            np.array([ix, iy, iz]).reshape(3, 1), k=nsample)
        inds = index[0]
        edge_id = []
        #  print(index)

        for ii in range(1, len(inds)):
            nx = sample_x[inds[ii]]
            ny = sample_y[inds[ii]]
            nz = sample_z[inds[ii]]

            if not is_collision(ix, iy, iz, nx, ny, nz, rr, obkdtree):
                edge_id.append(inds[ii])

            if len(edge_id) >= N_KNN:
                break

        road_map.append(edge_id)

    # plot_road_map(road_map, sample_x, sample_y)

    return road_map


def dijkstra_planning(sx, sy, sz, gx, gy, gz, ox, oy, oz, rr, road_map, sample_x, sample_y, sample_z):
    """
    sx: start x position [m]
    sy: start y position [m]
    gx: goal x position [m]
    gy: goal y position [m]
    ox: x position list of Obstacles [m]
    oy: y position list of Obstacles [m]
    rr: robot radius [m]
    road_map: ??? [m]
    sample_x: ??? [m]
    sample_y: ??? [m]

    @return: Two lists of path coordinates ([x1, x2, ...], [y1, y2, ...]), empty list when no path was found
    """
    # ax = plt.axes(projection='3d')
    nstart = Node(sx, sy, sz, 0.0, -1)
    ngoal = Node(gx, gy, gz, 0.0, -1)

    openset, closedset = dict(), dict()
    openset[len(road_map) - 2] = nstart
    path_found = True

    while True:
        # print(len(openset))
        if not openset:
            print("openset empty")
            path_found = False
            break

        c_id = min(openset, key=lambda o: openset[o].cost)
        current = openset[c_id]

        # show graph
        # if show_animation and len(closedset.keys()) % 2 == 0:
        #     # for stopping simulation with the esc key.
        #     # plt.gcf().canvas.mpl_connect('key_release_event',
        #     #         lambda event: [exit(0) if event.key == 'escape' else None])
        #     # ax.scatter3D(current.x, current.y, current.z, color='blue', s=10);
        #     ax.scatter(current.x, current.y, current.z, color='blue', marker="o");
        #     # .plot(current.x, current.y, "xg")
        #     # plt.pause(0.001)

        if c_id == (len(road_map) - 1):
            print("goal is found!")
            ngoal.pind = current.pind
            ngoal.cost = current.cost
            break

        # Remove the item from the open set
        del openset[c_id]
        # Add it to the closed set
        closedset[c_id] = current

        # expand search grid based on motion model
        for i in range(len(road_map[c_id])):
            n_id = road_map[c_id][i]
            dx = sample_x[n_id] - current.x
            dy = sample_y[n_id] - current.y
            dz = sample_z[n_id] - current.z
            d = math.sqrt(dx**2 + dy**2 + dz**2)
            node = Node(sample_x[n_id], sample_y[n_id], sample_z[n_id], current.cost + d, c_id)

            if n_id in closedset:
                continue
            # Otherwise if it is already in the open set
            if n_id in openset:
                if openset[n_id].cost > node.cost:
                    openset[n_id].cost = node.cost
                    openset[n_id].pind = c_id
            else:
                openset[n_id] = node

    if path_found is False:
        return [], [], []

    # generate final course
    rx, ry, rz = [ngoal.x], [ngoal.y], [ngoal.z]
    pind = ngoal.pind
    while pind != -1:
        n = closedset[pind]
        rx.append(n.x)
        ry.append(n.y)
        rz.append(n.z)
        pind = n.pind

    return rx, ry, rz


def plot_road_map(road_map, sample_x, sample_y):  # pragma: no cover

    for i, _ in enumerate(road_map):
        for ii in range(len(road_map[i])):
            ind = road_map[i][ii]
            ###########################################
            # ax.scatter3D(goal[0], goal[1], goal[2], color='green', s=10);
            # plt.plot([sample_x[i], sample_x[ind]],
            #          [sample_y[i], sample_y[ind]], "-k")


def sample_points(sx, sy, sz, gx, gy, gz, rr, ox, oy, oz, obkdtree):
    maxx = gx+3
    maxy = gy+3
    maxz = gz+3
    minx = sx-3
    miny = sy-3
    minz = sz-3

    sample_x, sample_y, sample_z = [], [], []

    while len(sample_x) <= N_SAMPLE:
        tx = (random.random() * (maxx - minx)) + minx
        ty = (random.random() * (maxy - miny)) + miny
        tz = (random.random() * (maxz - minz)) + minz

        index, dist = obkdtree.search(np.array([tx, ty, tz]).reshape(3, 1))

        if dist[0] >= rr:
            sample_x.append(tx)
            sample_y.append(ty)
            sample_z.append(tz)

    sample_x.append(sx)
    sample_y.append(sy)
    sample_z.append(sz)
    sample_x.append(gx)
    sample_y.append(gy)
    sample_z.append(gz)

    return sample_x, sample_y, sample_z

def move_obs(ox,oy,oz):
    ox[-250:-1]+=0.20
    oy[-250:-1]+=0.20
    oz[-250:-1]+=0.20
    ox[-500:-250]-=0.20
    oy[-500:-250]-=0.20
    oz[-500:-250]-=0.20
    return ox,oy,oz

def main():
    print(__file__ + " start!!")

    # start and goal position
    sx = 10.0  # [m]
    sy = 10.0  # [m]
    sz = 10.0
    gx = 50.0  # [m]
    gy = 50.0  # [m]
    gz = 50.0
    robot_size = 5.0  # [m]

    ox = []
    oy = []
    oz = []


    for i in range(60):
        for j in range(60):
            oy.append(i)
            ox.append(0.00)
            oz.append(j)
    for i in range(60):
        for j in range(60):
            oy.append(i)
            ox.append(60.00)
            oz.append(j)

    for i in range(60):
        for j in range(2,12):
            for k in range(20,25):
                oy.append(j)
                ox.append(i)
                oz.append(k)

    for i in range(60):
        for j in range(55,60):
            for k in range(40,50):
                oy.append(j)
                ox.append(i)
                oz.append(k)

    for i in range(20,30):
        for j in range(25,30):
            for k in range(20,25):
                oy.append(j)
                ox.append(i)
                oz.append(k)

    for i in range(30,40):
        for j in range(30,35):
            for k in range(35,40):
                oy.append(j)
                ox.append(i)
                oz.append(k)

    ox = np.array(ox).astype(float)
    oy = np.array(oy).astype(float)
    oz = np.array(oz).astype(float)

    while(1):
        ox,oy,oz = move_obs(ox,oy,oz)
        # print(ox[-1:-250])
        go = [gx,gy,gz]
        current_point1 = [sx,sy,sz]
        print(current_point1)
        dis = np.linalg.norm(np.asarray(go)-np.asarray(current_point1))
        print(dis)
        if(dis<5):
            print("Goal Reached")
            break
        # gx = ((ogx-sx)*15)/(dis) + sx
        # gy = ((ogy-sy)*15)/(dis) + sy
        # gz = ((ogz-sz)*15)/(dis) + sz
        plt.cla()
        if show_animation:
            ax.scatter(ox, oy, oz, color='g', marker = "o")
            ax.scatter(sx, sy, sz, color='r', marker = "^");
            ax.scatter(gx, gy, gz, color='r', marker = "^");
            u = np.linspace(0, np.pi, 10)
            v = np.linspace(0, 2 * np.pi, 10)
            x = sx+15*np.outer(np.sin(u), np.sin(v))
            y = sy+15*np.outer(np.sin(u), np.cos(v))
            z = sz+15*np.outer(np.cos(u), np.ones_like(v))
            ax.plot_wireframe(x, y, z, color="b")
        nox = []
        noy = []
        noz = []
        for i in range(len(ox)):
            obs = [ox[i],oy[i],oz[i]]
            current_point1 = [sx,sy,sz]
            if(np.linalg.norm(np.asarray(current_point1)-np.asarray(obs))<15):
                nox.append(ox[i])
                noy.append(oy[i])
                noz.append(oz[i])

        print(len(ox),len(nox))
        rx, ry, rz= PRM_planning(sx, sy, sz, gx, gy, gz, nox, noy, noz, robot_size)
        nx=rx[-2]
        ny=ry[-2]
        nz=rz[-2]
        sx = ((nx-sx)*15)/(dis) + sx
        sy = ((ny-sy)*15)/(dis) + sy
        sz = ((nz-sz)*15)/(dis) + sz
        if show_animation:
            ax.plot(np.array(rx),np.array(ry),np.array(rz))
            # print(rx)
            # ax.scatter3D(rx, ry, rz, color='red', s=10)
            ax.scatter(rx, ry, rz, color='r', marker = "o")
            # plt.figure()
            plt.draw()
            plt.pause(0.1)


if __name__ == '__main__':
    main()
