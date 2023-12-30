import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pdb
import os
import imageio
import time
import math

from models import kinematic_bicycle_model, obstacle_model
from reference_path import reference_segment, referenceGenerator
from plot_functions import plotGIF
from world import buildStaticWorld, goal_test, near_goal_test, collisionFreeSearch

# prepare gif folder for animation
folder_name = "mpc_imgfolder"
show_animation = True
max_iteration = 50

if not os.path.exists(folder_name):
    os.makedirs(folder_name)
else:
    # Get the list of files and subdirectories in the folder
    items = os.listdir(folder_name)

    # Remove files and subdirectories in the folder
    for item in items:
        item_path = os.path.join(folder_name, item)
        if os.path.isfile(item_path):
            os.remove(item_path)
        if os.path.isdir(item_path):
            os.rmdir(item_path)

# log file for optime
optimefile = 'execution_time.txt'
opfile = open(os.path.join(os.path.dirname(__file__), optimefile), 'w')


# add obstacle as constraints
def obstacle_constraint(u, *args):
    x0, dt, N, Q, R, reference, obstacles, obstacle_radius,vref,collisionfreeset, worldH, worldW = args
    constraints = []

    obss = np.copy(obstacles)
    x = np.copy(x0)
    for i in range(N):
        # Add obstacle avoidance constraints
        for obstacle in obss:
            distance = np.sqrt((x[0] - obstacle[0])**2 + (x[1] - obstacle[1])**2)
            constraints.append(distance - obstacle_radius - 0.1)  # Keep a small safety margin
            obstacle_model(obstacle, dt, worldW, worldH)

        # velocity constraint
        constraints.append(x[3] < 20)
        # static constraints
        if collisionfreeset[0] != collisionfreeset[2] and collisionfreeset[1] != collisionfreeset[3]:
            if x[2]<=np.pi/2:
                constraints.append(1/(collisionfreeset[2] - x[0]))
                constraints.append(1/(collisionfreeset[3] - x[1]))
            elif x[2]<=np.pi:
                constraints.append(1 / (collisionfreeset[3] - x[1]))
                constraints.append(1 / (x[0] - collisionfreeset[0]))
            elif x[2]<=np.pi*3/2:
                constraints.append(1 / (x[1] - collisionfreeset[1]))
                constraints.append(1 / (x[0] - collisionfreeset[0]))
            else:
                constraints.append(1 / (collisionfreeset[2] - x[0]))
                constraints.append(1 / (x[1] - collisionfreeset[1]))
        else:
            constraints.append(5)
            constraints.append(5)

        u_i = [u[2*i], u[2*i+1]]
        x_i = np.copy(x)
        kinematic_bicycle_model(x_i, u_i, dt)
        x = x_i
    return np.array(constraints)


# Define the objective function for MPC
def objective_function(u, *args):
    #print(args)
    x0, dt, N, Q, R, reference, obstacles, obstacle_radius, vref,collisionfreeset,worldH, worldW = args
    cost = 0.0
    x = np.copy(x0)

    for i in range(N):
        phi0 = 0
        try:
            phi0 = np.arctan((reference[i+1][1]-reference[i][1])/(reference[i+1][0]-reference[i][0]))
        except IndexError:
            phi0 = np.arctan((reference[-1][1]-reference[0][1])/(reference[-1][0]-reference[0][0]))

        phi = x[2] - phi0
        deltav = x[3] - vref
        cost += (x[:2] - reference[i][:2]).T @  Q  @ (x[:2] - reference[i][:2])  # Tracking cost
        cost += 3*phi**2
        cost += 4*deltav**2
        # obstacle as soft constraint
        for obstacle in obstacles:
            cost += 2*np.exp(-((x[0] - obstacle[0])**2 + (x[1] - obstacle[1])**2) / (2 * (obstacle_radius+0.5)**2))

        u_i = [u[2*i], u[2*i+1]]
        x_i = np.copy(x)
        kinematic_bicycle_model(x_i, u_i, dt)
        x = x_i

    cost += (x[:2] - reference[N][:2]).T @ Q @ (x[:2] - reference[N][:2]) # Terminal tracking cost
    #print(cost)
    for i in range(N):
        u_i = np.array([[u[2*i],], [u[2*i+1],]])
        cost += u_i.T @ R @ u_i  # Control cost
    
    return cost

# MPC Path Tracking with Obstacle Avoidance
def mpc_path_tracking(x0, dt, N, Q, R, reference, obstacles, obstacle_radius, vref,collisionfreeset, worldH, worldW):
    u0 = np.zeros((N, 2))  # Initial guess for control inputs

    bounds = [(-np.radians(60), np.radians(60)), (-5.0, 5.0)]  # Control input bounds

    # without obstacle constraints
    # result = minimize(objective_function, u0.flatten(), args=(x0, dt, N, Q, R, reference, obstacles, obstacle_radius, vref),
    #                   bounds=bounds * N, method='SLSQP')

    # Define obstacle constraints
    obstacle_constraints = ({'type': 'ineq', 'fun': obstacle_constraint, 'args': (x0, dt, N, Q, R, reference, obstacles, obstacle_radius,vref,collisionfreeset, worldH,worldW)})
    result = minimize(objective_function, u0.flatten(), args=(x0, dt, N, Q, R, reference, obstacles, obstacle_radius, vref,collisionfreeset,worldH,worldW),
                      bounds=bounds * N, constraints=obstacle_constraints, method='SLSQP')
    u_optimal = result.x.reshape((N, 2))

    return u_optimal


def main():
    # Simulation parameters
    dt = 0.1
    N = 5  # MPC horizon
    NrefTotal = 10 # total amount of reference points
    refAmount = 10 # amount of reference segments
    Q = np.diag([1.0, 1.0])  # Tracking cost matrix
    R = np.diag([0.0, 0.0])  # Control cost matrix
    maxR = 5 # Search Distance

    # Initial state
    x0 = np.array([0.0, 0.0, 0, 5.0])
    real_trajectory = [np.copy(x0)]
    vref = 5

    # build static map
    columnCorners = np.array([[5,4], [14,9]])
    worldH, worldW = 30, 30
    staticScene = buildStaticWorld(columnCorners, worldH, worldW)

    # Only for testing purpose
    if (0):
        # collisionTest
        xtest = np.array([[1.0, 1.0, 0.5, 5.0],
                        [1.0, 2.0, 0.6, 5.0],
                        [2.0, 2.0, 0.9, 5.0],
                        [3.0, 3.0, 1.0, 5.0],
                        [4.0, 4.0, 1.0, 5.0],
                        [2.0, 7.0, 1.0, 5.0],
                        [7.0, 2.0, 1.0, 5.0],
                        [8.0, 8.0, 1.0, 5.0]])
        # discretize x and y
        for iX in range(xtest.shape[0]):
            print("curr loc: ", xtest[iX])
            discrX0 = np.zeros((2,), dtype=int)
            discrX0[0], discrX0[1] = int(xtest[iX][0]), int(xtest[iX][1])
            theta = xtest[iX][2]
            xMin, xMax, yMin, yMax = collisionFreeSearch(discrX0, theta*math.pi, maxR, staticScene)
            
            # getRotateOccupancy( 7, 11, 2, 0, -90/180*math.pi, staticScene)
            # collisionCheck( [7, 11], staticScene)

            xMin += xtest[iX][0]
            xMax += xtest[iX][0]
            yMin += xtest[iX][1]
            yMax += xtest[iX][1]
            collisionFreeSet = [xMin, yMin, xMax, yMax]
            fig, ax = plt.subplots()
            plt.imshow(1 - staticScene, cmap='gray', origin='lower')
            # show collision free region
            x_min = collisionFreeSet[0]
            y_min = collisionFreeSet[1]
            x_max = collisionFreeSet[2]
            y_max = collisionFreeSet[3]
            # rectangle = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, edgecolor='pink', facecolor='none')
            # plt.gca().add_patch(rectangle)
            center = [x_min + (x_max - x_min)/2, y_min + (y_max - y_min)/2]
            width, height = x_max - x_min, y_max - y_min
            angle_degrees = xtest[iX][2] * 180
            draw_rotated_rectangle(ax, center, width, height, angle_degrees)
            plt.plot(discrX0[0], discrX0[1],"xg")
            plt.savefig(f"tmp{iX}.png")
            # plt.show()
            plt.close()

    # Set random seed for debugging
    np.random.seed(10)

    # Reference trajectory
    reference = np.vstack([np.array([i, i]) for i in range(NrefTotal + 1)])
    reference_delV = x0[3]
    # seperate reference into segments
    refSegments = reference_segment(reference, refAmount)

    # Obstacles
    v = 3
    theta = -3*np.pi/4
    # Obstacle numbers
    obstacles_num = 10
    obstacles = np.zeros((obstacles_num, 4))
    for i in range(obstacles_num):
        obstacles[i, 0] = np.random.uniform(1, 15)  # x
        obstacles[i, 1] = np.random.uniform(1, 15)  # y
        obstacles[i, 2] = np.random.uniform(0, 2 * np.pi)  # theta
        obstacles[i, 3] = np.random.uniform(0.1, 1)  # v
    # Obstacle radius
    obstacle_radius = 0.4
    # Obstacle trajectory
    obsts = []
    real_obstacle_traj = []
    for j in range(len(obstacles)):
        real_obstacle_traj.append(np.copy(obstacles[j])[np.newaxis,:])
        obstrajectory = [np.copy(obstacles[j])]
        for i in range(N):
            o_i = np.copy(obstrajectory[-1])
            obstacle_model(o_i, dt,worldW,worldH)
            obstrajectory.append(o_i)
        obsts.append(np.array(obstrajectory))
    # print("obsts: ", obsts)
    
    # Run MPC
    for iter in range(max_iteration):
        if goal_test(x0, reference[-1]):
            print("triggered goal")
            break

        if near_goal_test(x0, reference[-1]):
            #print("triggered near")
            vref = 0

        starttime = time.time()
        # discretize x and y
        discrX0 = np.zeros((2,), dtype=int)
        discrX0[0], discrX0[1] = int(x0[0]), int(x0[1])
        theta = x0[2]
        xMin, xMax, yMin, yMax = collisionFreeSearch(discrX0, theta, maxR, staticScene)
        xMin += x0[0]
        xMax += x0[0]
        yMin += x0[1]
        yMax += x0[1]
        collisionFreeSet = [xMin, yMin, xMax, yMax]

        referenceForMPC = referenceGenerator(refSegments, reference[-1],reference_delV, x0, N, dt)
        u_optimal = mpc_path_tracking(x0, dt, N, Q, R, referenceForMPC, obstacles, obstacle_radius, vref, collisionFreeSet, worldH,worldW)
        endtime = time.time()
        optime = endtime - starttime
        opfile.write(str(optime) + "\n")
        #print("u opt: ",u_optimal)

        # Simulate the system
        MPC_trajectory = [np.copy(x0)]
        for i in range(N):
            x_i = np.copy(MPC_trajectory[-1])
            kinematic_bicycle_model(x_i, u_optimal[i], dt)
            MPC_trajectory.append(x_i)

        # Plot the results
        MPC_trajectory = np.array(MPC_trajectory)
        if show_animation:  # pragma: no cover
            plotGIF(iter, reference, MPC_trajectory, real_trajectory, referenceForMPC, obsts, real_obstacle_traj, staticScene, collisionFreeSet, theta)

        # agent's next state depending on u_optimal[0]
        u_mpc = u_optimal[0]
        kinematic_bicycle_model(x0, u_mpc, dt)
        real_trajectory.append(np.copy(x0))

        # obstacles' states update
        for j in range(len(obstacles)):
            obstacle_model(obstacles[j], dt, worldW,worldH)
            real_obstacle_traj[j] = np.vstack((real_obstacle_traj[j], obstacles[j]))


if __name__ == '__main__':
    main()
    if show_animation and os.path.exists(folder_name):
        # Create a GIF from the saved figures
        image_files = [os.path.join(folder_name,file) for file in os.listdir(folder_name) if os.path.isfile(os.path.join(folder_name, file))]
        output_gif_path = os.path.join(folder_name, 'output.gif')
        with imageio.get_writer(output_gif_path, duration=0.5) as writer:
            for filename in image_files:
                image = imageio.imread(filename)
                writer.append_data(image)
            # Remove files in the folder
        for item_path in image_files:
            if os.path.isfile(item_path):
                os.remove(item_path)
