import numpy as np
import os
import imageio
import time
from scipy.optimize import minimize

from models import kinematic_bicycle_model, obstacle_model
from reference_path import reference_segment, referenceGenerator
from plot_functions import plotGIF
from control_policy import config, RobotType, dwa_control


# gif folder for plt
folder_name = "dw_imgfolder"
show_animation = True
max_iteration = 30

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
optimefile = 'execution_time_dw.txt'
opfile = open(os.path.join(os.path.dirname(__file__), optimefile), 'w')


def goal_test(curState, target):
    curPos = curState[:2]
    if np.linalg.norm(curPos - target) < 0.7:
        return True
    return False

def near_goal_test(curState, target):
    curPos = curState[:2]
    if np.linalg.norm(curPos - target) < 5:
        return True
    return False

def pos_collision_check(pos, obs, obstacle_radius):
    count = 0
    collision_velocities = []
    for i in range(len(obs)):
        obstacle = obs[i]
        distance = np.sqrt((pos[0] - obstacle[0])**2 + (pos[1] - obstacle[1])**2)
        if (distance - obstacle_radius - 0.55 < 0):
            count += 1
            collision_velocities.append(pos[3])
    return count, collision_velocities

def static_obstacles(N):
    obs = []
    for i in range(N):
        for j in range(N):
            if i == 0 or j == 0 or i == N-1 or j == N-1:
                obs.append([i,j,0,0])
    return obs


def main(gx=10.0, gy=10.0, robot_type=RobotType.circle):
    # Simulation parameters
    dt = 0.1
    N = 5  # MPC horizon
    NrefTotal = 20 # total amount of reference points
    refAmount = 10 # amount of reference segments

    # Initial state
    x0 = np.array([0.0, 0.0, 0.0, 5.0, 0.0])
    config.robot_type = robot_type
    real_trajectory = [np.copy(x0)]
    goal = np.array([gx, gy])
    vref = 5
    collisions = 0
    collision_velocities = []
    iterations = 0

    # Set random seed for debugging
    np.random.seed(80)

    # Reference trajectory
    reference = np.vstack([np.array([i, i]) for i in range(NrefTotal + 1)])

    # Obstacles
    v = 3
    theta = -3*np.pi/4
    # Obstacle numbers
    obstacles_num = 4
    obstacles = np.zeros((obstacles_num, 4))
    for i in range(obstacles_num):
        obstacles[i, 0] = np.random.uniform(0, 15)  # x
        obstacles[i, 1] = np.random.uniform(0, 15)  # y
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
            obstacle_model(o_i, dt)
            obstrajectory.append(o_i)
        obsts.append(np.array(obstrajectory))
    # obstacles need to update every time frame
    config.ob = obstacles
    static_obs = static_obstacles(NrefTotal)

    # Run MPC
    for iter in range(max_iteration):
        if goal_test(x0, goal):
            print("triggered goal")
            break
        
        if near_goal_test(x0, goal):
            print("triggered near")
            vref = 0

        iterations += 1
        total_obs = np.copy(obstacles)
        for i in range(len(static_obs)):
            np.append(total_obs, static_obs[i])

        starttime = time.time()
        reference_delV = x0[3]
        # seperate reference into segments
        refSegments = reference_segment(reference, refAmount)
        referenceForMPC = referenceGenerator(refSegments, reference[-1],reference_delV, x0, N, dt)
        u_optimal, predicted_trajectory = dwa_control(x0, config, referenceForMPC[-1], total_obs, vref)
        endtime = time.time()
        optime = endtime - starttime
        opfile.write(str(optime) + "\n")
        print("u opt: ",u_optimal)

        coln, colv = pos_collision_check(x0, obstacles, obstacle_radius)
        collisions += coln
        collision_velocities += colv

        if show_animation:  # pragma: no cover
            plotGIF(iter, reference, predicted_trajectory, real_trajectory, obsts, real_obstacle_traj, referenceForMPC)

        # agent's next state depending on u_optimal[0]
        u_mpc = u_optimal
        x0 = kinematic_bicycle_model(x0, u_mpc, dt)
        real_trajectory.append(np.copy(x0))

        # obstacles' states update
        for j in range(len(obstacles)):
            obstacle_model(obstacles[j], dt)
            real_obstacle_traj[j] = np.vstack((real_obstacle_traj[j], obstacles[j]))

    opfile.write("total iterations: " + str(iterations) + "\n")
    opfile.write("collisions: " + str(collisions) + "\n")
    opfile.write("collisions velocities mean: " + str(np.mean(collision_velocities)) + "\n")


if __name__ == '__main__':
    main(20, 20, robot_type=RobotType.circle)
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