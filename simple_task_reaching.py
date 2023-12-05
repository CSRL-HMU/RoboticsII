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

# Create the robot (both for Leader and Follower)
ur = rt.models.UR5()

# Get initial configuration
q0 = np.array(rtde_r.getActualQ())

# Get initial end-eefector pose
g0 = ur.fkine(q0)
R0 = np.array(g0.R)
p0 = np.array(g0.t)


# Control cycle
dt = 0.002

# Init time
t = 0.0

# Start logging
plog = p0
tlog = t

# get time now
t_now = time.time()

# initialize qdot
qdot = np.zeros(6)


# ------------------ For the controller

# Target in joint space
pT = p0 + [0.2, - 0.03, 0.4]
pTlog = pT

# Gains
K = 0.5 * np.identity(3)

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

    # Get  end-eefector pose
    g = ur.fkine(q)
    R = np.array(g.R)
    p = np.array(g.t)

    # get full jacobian
    J = np.array(ur.jacob0(q))

    # get translational jacobian
    Jp = J[:3]

    # pseudoInverse
    JpInv = np.linalg.pinv(Jp)

    # reaching control signal
    qdot = - JpInv @ K @ (p - pT)


    # set joint speed
    rtde_c.speedJ(qdot, 20.0, dt)

    # log data
    tlog = np.vstack((tlog, t))
    plog = np.vstack((plog, p))
    pTlog = np.vstack((pTlog, pT))

    # synchronize
    rtde_c.waitPeriod(t_start)

# close control
rtde_c.speedStop()
rtde_c.stopScript()


# plot results
fig, axs = plt.subplots(3)
fig.suptitle('Joint tracking')
for i in range(3):
    axs[i].plot(tlog, plog[:, i], 'k', linewidth=2.0)
    axs[i].plot(tlog, pTlog[:, i], 'r--', linewidth=2.0)


plt.show()
