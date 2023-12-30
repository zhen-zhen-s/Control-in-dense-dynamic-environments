import numpy as np
import math

# Define the object as bicycle model
def kinematic_bicycle_model(x, u, dt):
    """
    Kinematic bicycle model equations.
    x: state vector [x, y, theta, v]
    u: control vector [delta, a]
    dt: time step
    """
    L = 0.5  # Wheelbase
    x[2] += u[1] * dt
    x[0] += u[0] * math.cos(x[2]) * dt
    x[1] += u[0] * math.sin(x[2]) * dt
    x[3] = u[0]
    x[4] = u[1]
    return x

# add obstacle dynamics model
def obstacle_model(x, dt, xbound = 20, ybound=20):

    # Calculate new position
    newx = x[3] * np.cos(x[2]) * dt + x[0]
    newy = x[3] * np.sin(x[2]) * dt + x[1]

    # Check if within bound
    if newx >= 0 and newx <= xbound and newy >= 0 and newy <= ybound:
        # Move
        x[0], x[1] = newx, newy
    # Else do nothing
