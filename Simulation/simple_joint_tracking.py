import roboticstoolbox as rt
import numpy as np
import scipy as sp
import spatialmath as sm
from tqdm import tqdm
import matplotlib.pyplot as plt
import math
import time
import keyboard


# Declare math pi
pi = math.pi


# commanded initial configuration for testing
q0c = np.array([-30, -113, -97, -73,  100, -73])
q0c = q0c * pi / 180.0  # degrees to rad



# Get initial configuration
q0 = q0c
q = q0


# Create the robot (both for Leader and Follower)
ur = rt.models.UR5()

# Control cycle
dt = 0.002

# Init time
t = 0.0

# Start logging
qlog = q0

tlog = t

# get time now
t_now = time.time()

# initialize qdot
qdot = np.zeros(6)



# ------------------ For the controller

# Trajectory initialization in joint space
qd0 = q0 + 0.05*np.ones(6)
qd0[1] = qd0[1] - 0.25
qdlog = qd0


# trajectory frequency
f1 = 0.7
f2 = 0.3

# Gains
K = 10.0 * np.identity(6)

# -------------------------------------


for i in range(5000):

    if keyboard.is_pressed('a'):
        print('Stopping robot')
        break

    # print(time.time() - t_now)
    t_now = time.time()



    # print(actual_q)


    qd = qd0 + [(pi / 8) * math.sin(2 * pi * f1 * t), (pi / 6) * math.sin(2 * pi * f2 * t), 1 - math.exp(-0.5 * t),
               -(pi / 3) * math.sin(2 * pi * f2 * t), 0.5 * (pi / 3) * math.sin(2 * pi * f2 * t),
               -(pi / 4) * math.sin(2 * pi * f1 * t)]

    qddot = [2 * pi * f1 * (pi / 8) * math.cos(2 * pi * f1 * t), 2 * pi * f2 * (pi / 6) * math.cos(2 * pi * f2 * t), 0.5 * math.exp(-0.5 * t),
               - 2 * pi * f2 * (pi / 3) * math.cos(2 * pi * f2 * t), 2 * pi * f2 * 0.5 * (pi / 3) * math.cos(2 * pi * f2 * t),
               - 2 * pi * f1 * (pi / 4) * math.cos(2 * pi * f1 * t)]

    # Integrate time
    t = t + dt

    # Get joint values
    q = q + qdot * dt


    # reaching control signal
    qdot = qddot - K @ (q - qd)



    # log data
    tlog = np.vstack((tlog, t))
    qlog = np.vstack((qlog, q))
    qdlog = np.vstack((qdlog, qd))




# plot results
fig, axs = plt.subplots(6)
fig.suptitle('Joint tracking')
for i in range(6):
    axs[i].plot(tlog, qlog[:, i], 'k', linewidth=2.0)
    axs[i].plot(tlog, qdlog[:, i], 'r--', linewidth=2.0)


plt.show()

