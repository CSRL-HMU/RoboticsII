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

# Target in joint space
qT = np.array([-45, -70, -30, -50,  80, 20])
qT = qT * pi / 180.0  # degrees to rad
qTlog = qT

# Gains
K = 0.5 * np.identity(6)

# -------------------------------------


for i in range(5000):

    if keyboard.is_pressed('a'):
        print('Stopping robot')
        break

    # print(time.time() - t_now)
    t_now = time.time()
    

    # print(actual_q)


    # Integrate time
    t = t + dt

    # Get joint values
    q = q + qdot * dt


    # reaching control signal
    qdot = - K @ (q - qT)



    # log data
    tlog = np.vstack((tlog, t))
    qlog = np.vstack((qlog, q))
    qTlog = np.vstack((qTlog, qT))



print('Heelo')


# plot results
fig, axs = plt.subplots(6)
fig.suptitle('Joint tracking')
for i in range(6):
    axs[i].plot(tlog, qlog[:, i], 'k', linewidth=2.0)
    axs[i].plot(tlog, qTlog[:, i], 'r--', linewidth=2.0)


plt.show()
