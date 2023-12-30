import numpy as np

# Define the object as bicycle model
def kinematic_bicycle_model(x, u, dt):
    """
    Kinematic bicycle model equations.
    x: state vector [x, y, theta, v]
    u: control vector [delta, a]
    dt: time step
    """
    L = 2.5  # Wheelbase
    x[0] += x[3] * np.cos(x[2]) * dt
    x[1] += x[3] * np.sin(x[2]) * dt
    x[2] += (x[3] / L) * np.tan(u[0]) * dt
    x[3] += u[1] * dt

# add obstacle dynamics model
def obstacle_model(x, dt, xbound, ybound):

    # Calculate new position
    newx = x[3] * np.cos(x[2]) * dt + x[0]
    newy = x[3] * np.sin(x[2]) * dt + x[1]

    # Check if within bound
    if newx >= 0 and newx <= xbound and newy >= 0 and newy <= ybound:
        # Move
        x[0], x[1] = newx, newy
    # Else do nothing
