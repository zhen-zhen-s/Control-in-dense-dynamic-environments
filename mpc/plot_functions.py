import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
import os

# Vehicle parameters
LENGTH = 1.5  # [m]
WIDTH = 0.7  # [m]
BACKTOWHEEL = 0.3  # [m]
WHEEL_LEN = 0.1  # [m]
WHEEL_WIDTH = 0.03  # [m]
TREAD = 0.2  # [m]
WB = 0.8  # [m]
folder_name = "mpc_imgfolder"

def plotGIF(iter, reference, MPC_trajectory, real_trajectory, referenceForMPC, obsts, real_obstacle_traj, staticScene,
            collisionFreeSet, theta):
    obsradius = 0.4
    th = np.arange(0, 2 * np.pi, 0.01)
    for idx in range(len(MPC_trajectory)):

        plt.cla()
        fig, ax = plt.subplots()
        plt.xlim(np.min(reference), np.max(reference))
        plt.ylim(np.min(reference), np.max(reference))

        # for stopping simulation with the esc key.
        # plt.gcf().canvas.mpl_connect('key_release_event',
        #         lambda event: [exit(0) if event.key == 'escape' else None])

        # show static scene
        plt.imshow(1 - staticScene, cmap='gray', origin='lower')
        # show collision free region
        x_min = collisionFreeSet[0]
        y_min = collisionFreeSet[1]
        x_max = collisionFreeSet[2]
        y_max = collisionFreeSet[3]

        # Draw the rotated rectangle
        center = [x_min + (x_max - x_min) / 2, y_min + (y_max - y_min) / 2]
        width, height = x_max - x_min, y_max - y_min
        draw_rotated_rectangle(ax, center, width, height, theta)

        # show predicted optimal trajectory in each iteration
        plt.plot(MPC_trajectory[:idx, 0], MPC_trajectory[:idx, 1], "-", color='lightblue', label="MPC trajectory")
        # show trajectory record
        plt.plot(np.array(real_trajectory)[:, 0], np.array(real_trajectory)[:, 1], "-", color='blue',
                 label="real trajectory")
        # show reference path fed into MPC
        plt.plot(referenceForMPC[:idx, 0], referenceForMPC[:idx, 1], "--", color="orange", label="ref path")

        # show multiple obstacles' trajectory record
        for i in range(len(real_obstacle_traj)):
            plt.plot(real_obstacle_traj[i][:, 0], real_obstacle_traj[i][:, 1], '-', label='Obstacles')
            # plt.plot(obsts[i][:idx, 0], obsts[i][:idx, 1], '-', label='Obstacles')
            a = real_obstacle_traj[i][-1, 0] + obsradius * np.cos(th)
            b = real_obstacle_traj[i][-1, 1] + obsradius * np.sin(th)
            plt.plot(a, b, linestyle='-')

        plt.plot(reference[-1, 0], reference[-1, 1], "xg", label="target")
        # print(np.array(real_trajectory).ndim)
        plot_car(np.array(real_trajectory)[-1, 0], np.array(real_trajectory)[-1, 1], np.array(real_trajectory)[-1, 2])
        plt.axis("equal")
        plt.grid(True)
        plt.legend()
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title("ITER:" + str(iter) + ", MPC n:" + str(idx))
        # plt.title("Time[s]:" + str(round(time, 2)) + ", speed[km/h]:" + str(round(state.v * 3.6, 2)))

        # save figures in one folder
        plt.savefig(os.path.join(folder_name, f'figure_{(iter * len(MPC_trajectory) + idx):03d}.png'))
        print("idx: ", idx, " iter*len(MPC_trajectory): ", iter * len(MPC_trajectory),
              " (iter*len(MPC_trajectory)+idx): ", (iter * len(MPC_trajectory) + idx))
        # plt.pause(0.0001)
        plt.close()

# Function to draw a rotated rectangle
def draw_rotated_rectangle(ax, center, width, height, rad, color='blue'):
    # Create a rotated rectangle using Affine2D transformation
    angleDegree = rad / math.pi *180
    rectangle = matplotlib.patches.Rectangle((-width/2, -height/2), width, height, color=color, alpha=0.5)
    transform = matplotlib.transforms.Affine2D().rotate_deg(angleDegree).translate(*center)
    rectangle.set_transform(transform + ax.transData)

    # Add the rectangle to the plot
    ax.add_patch(rectangle)

def plot_car(x, y, yaw, steer=0.0, cabcolor="-r", truckcolor="-k"):  # pragma: no cover

    outline = np.array([[-BACKTOWHEEL, (LENGTH - BACKTOWHEEL), (LENGTH - BACKTOWHEEL), -BACKTOWHEEL, -BACKTOWHEEL],
                        [WIDTH / 2, WIDTH / 2, - WIDTH / 2, -WIDTH / 2, WIDTH / 2]])

    fr_wheel = np.array([[WHEEL_LEN, -WHEEL_LEN, -WHEEL_LEN, WHEEL_LEN, WHEEL_LEN],
                         [-WHEEL_WIDTH - TREAD, -WHEEL_WIDTH - TREAD, WHEEL_WIDTH - TREAD, WHEEL_WIDTH - TREAD, -WHEEL_WIDTH - TREAD]])
    rr_wheel = np.copy(fr_wheel)
    fl_wheel = np.copy(fr_wheel)
    fl_wheel[1, :] *= -1
    rl_wheel = np.copy(rr_wheel)
    rl_wheel[1, :] *= -1

    Rot1 = np.array([[math.cos(yaw), math.sin(yaw)],
                     [-math.sin(yaw), math.cos(yaw)]])
    Rot2 = np.array([[math.cos(steer), math.sin(steer)],
                     [-math.sin(steer), math.cos(steer)]])

    fr_wheel = (fr_wheel.T.dot(Rot2)).T
    fl_wheel = (fl_wheel.T.dot(Rot2)).T
    fr_wheel[0, :] += WB
    fl_wheel[0, :] += WB
    fr_wheel = (fr_wheel.T.dot(Rot1)).T
    fl_wheel = (fl_wheel.T.dot(Rot1)).T
    outline = (outline.T.dot(Rot1)).T
    rr_wheel = (rr_wheel.T.dot(Rot1)).T
    rl_wheel = (rl_wheel.T.dot(Rot1)).T

    outline[0, :] += x
    outline[1, :] += y
    fr_wheel[0, :] += x
    fr_wheel[1, :] += y
    rr_wheel[0, :] += x
    rr_wheel[1, :] += y
    fl_wheel[0, :] += x
    fl_wheel[1, :] += y
    rl_wheel[0, :] += x
    rl_wheel[1, :] += y

    plt.plot(np.array(outline[0, :]).flatten(),
             np.array(outline[1, :]).flatten(), truckcolor)
    plt.plot(np.array(fr_wheel[0, :]).flatten(),
             np.array(fr_wheel[1, :]).flatten(), truckcolor)
    plt.plot(np.array(rr_wheel[0, :]).flatten(),
             np.array(rr_wheel[1, :]).flatten(), truckcolor)
    plt.plot(np.array(fl_wheel[0, :]).flatten(),
             np.array(fl_wheel[1, :]).flatten(), truckcolor)
    plt.plot(np.array(rl_wheel[0, :]).flatten(),
             np.array(rl_wheel[1, :]).flatten(), truckcolor)
    plt.plot(x, y, "*")