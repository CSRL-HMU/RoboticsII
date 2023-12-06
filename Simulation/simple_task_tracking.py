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
q0c = np.array([0, -113, -97, -73,  100, -73])
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

pd0 = p0
pd0[1] = pd0[1] + 0.02
pdlog = pd0

# trajectory frequency
f1 = 0.7
f2 = 0.3

# Gains
K = 10.0 * np.identity(3)

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



    pd = pd0 + [- 0.03 * math.sin(2 * pi * f2 * t),
                0.15 * math.sin(2 * pi * f1 * t),
                0.5 - 0.5 * math.exp(-0.5 * t)]

    pddot = [- 2 * pi * f2 * 0.03 * math.cos(2 * pi * f2 * t),
             2 * pi * f1 * 0.15 * math.cos(2 * pi * f1 * t),
             0.5 * 0.5 * math.exp(-0.5 * t)]

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

    # Null space projection matrix
    N = np.identity(6) - JpInv @ Jp

    # Velocity in the null space 
    qndot = math.sin(2 * 0.1 * pi * t) * (pi / 6) * np.ones(6)

    # reaching control signal
    qdot = JpInv @ (pddot - K @ (p - pd)) + N @ qndot



    # log data
    tlog = np.vstack((tlog, t))
    plog = np.vstack((plog, p))
    pdlog = np.vstack((pdlog, pd))





# plot results
fig, axs = plt.subplots(3)
fig.suptitle('Joint tracking')
for i in range(3):
    axs[i].plot(tlog, plog[:, i], 'k', linewidth=2.0)
    axs[i].plot(tlog, pdlog[:, i], 'r--', linewidth=2.0)


plt.show()
