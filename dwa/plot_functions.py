import numpy as np
import matplotlib.pyplot as plt
import os

folder_name = "dw_imgfolder"

def plotGIF(iter, reference, predicted_trajectory, real_trajectory, obsts, real_obstacle_traj, referenceForMPC):
    for idx in range(len(predicted_trajectory)):
        plt.cla()
        plt.xlim(np.min(reference), np.max(reference))
        plt.ylim(np.min(reference), np.max(reference))

        # for stopping simulation with the esc key.
        # plt.gcf().canvas.mpl_connect('key_release_event',
        #         lambda event: [exit(0) if event.key == 'escape' else None])
        # plt.plot(MPC_trajectory[:idx, 0], MPC_trajectory[:idx, 1], "-", color='lightblue', label="MPC trajectory")
        plt.plot(predicted_trajectory[:, 0], predicted_trajectory[:, 1], "-g")
        plt.plot(np.array(real_trajectory)[:, 0], np.array(real_trajectory)[:, 1], "-", color='blue',
                 label="real trajectory")
        plt.plot(referenceForMPC[:idx, 0], referenceForMPC[:idx, 1], "--", color="orange", label="ref path")
        for i in range(len(obsts)):
            plt.plot(obsts[i][:idx, 0], obsts[i][:idx, 1], '-', label='Obstacles')
        for i in range(len(real_obstacle_traj)):
            plt.plot(real_obstacle_traj[i][:, 0], real_obstacle_traj[i][:, 1], '-', label='Obstacles')
        plt.plot(reference[-1, 0], reference[-1, 1], "xg", label="target")
        # plot_car(state.x, state.y, state.yaw, steer=di)
        plt.axis("equal")
        plt.grid(True)
        plt.legend()
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title("ITER:" + str(iter) + ", MPC n:" + str(idx))
        # plt.title("Time[s]:" + str(round(time, 2)) + ", speed[km/h]:" + str(round(state.v * 3.6, 2)))
        # save figures in one folder
        plt.savefig(os.path.join(folder_name, f'figure_{(iter * len(predicted_trajectory) + idx):03d}.png'))
        # print("idx: ", idx, " iter*len(predicted_trajectory): ", iter*len(predicted_trajectory)," (iter*len(predicted_trajectory)+idx): ", (iter*len(predicted_trajectory)+idx))
        # plt.pause(0.0001)
        plt.close()