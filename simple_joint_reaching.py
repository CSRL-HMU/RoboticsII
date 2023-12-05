import roboticstoolbox as rt
import numpy as np
import scipy as sp
import spatialmath as sm
import rtde_receive
import rtde_control
from tqdm import tqdm
import matplotlib.pyplot as plt
import math
import time
import keyboard


# Declare math pi
pi = math.pi

# Define follower (UR5e)
rtde_c = rtde_control.RTDEControlInterface("192.168.1.100")
rtde_r = rtde_receive.RTDEReceiveInterface("192.168.1.100")



# commanded initial configuration for testing
q0c = np.array([-30, -113, -97, -73,  100, -73])
q0c = q0c * pi / 180.0  # degrees to rad

# Move to the initial configuration
rtde_c.moveJ(q0c, 0.5, 0.5)


# Get initial configuration
q0 = np.array(rtde_r.getActualQ())


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
    t_start = rtde_c.initPeriod()


    # print(actual_q)


    # Integrate time
    t = t + dt

    # Get joint values
    q = np.array(rtde_r.getActualQ())


    # reaching control signal
    qdot = - K @ (q - qT)


    # set joint speed
    rtde_c.speedJ(qdot, 20.0, dt)

    # log data
    tlog = np.vstack((tlog, t))
    qlog = np.vstack((qlog, q))
    qTlog = np.vstack((qTlog, qT))

    # synchronize
    rtde_c.waitPeriod(t_start)

# close control
rtde_c.speedStop()
rtde_c.stopScript()


# plot results
fig, axs = plt.subplots(6)
fig.suptitle('Joint tracking')
for i in range(6):
    axs[i].plot(tlog, qlog[:, i], 'k', linewidth=2.0)
    axs[i].plot(tlog, qTlog[:, i], 'r--', linewidth=2.0)


plt.show()
