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

# Move to the initial configuration
q0 = q0c
q = q0

# Create the robot (both for Leader and Follower)
ur = rt.models.UR5()


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



    # print(actual_q)


    # Integrate time
    t = t + dt

    # Get joint values
    q = q + qdot * dt

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



    # log data
    tlog = np.vstack((tlog, t))
    plog = np.vstack((plog, p))
    pTlog = np.vstack((pTlog, pT))





# plot results
fig, axs = plt.subplots(3)
fig.suptitle('Joint tracking')
for i in range(3):
    axs[i].plot(tlog, plog[:, i], 'k', linewidth=2.0)
    axs[i].plot(tlog, pTlog[:, i], 'r--', linewidth=2.0)


plt.show()
