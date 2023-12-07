import roboticstoolbox as rt
import numpy as np
import scipy as sp
import spatialmath as sm
from tqdm import tqdm
import matplotlib.pyplot as plt
import math
import time
import scienceplots

from matplotlib.pyplot import figure


plt.style.use(["default","no-latex"])


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





# plot results
fig = plt.figure(figsize=(4, 6))


for i in range(6):
    axs = fig.add_axes([0.21, ((5-i)/6)*0.9+0.1, 0.7, 0.12])
    axs.plot(tlog, qlog[:, i], 'k', linewidth=1.0)
    axs.plot(tlog, qTlog[:, i], 'k--', linewidth=2.0)
    axs.set_xlim([0, 10])
    axs.set_ylabel('$q_' + str(i+1) + '(t)$',fontsize=14 )
    
    if i==5:
        axs.set_xlabel('Time (s)',fontsize=14 )
        lgnd = axs.legend(['$q(t)$','$q_T(t)$'],fontsize=9,ncol=2,loc="upper right")
        lgnd.get_frame().set_alpha(None)
    else:
        axs.set_xticks([])


    
plt.show()
