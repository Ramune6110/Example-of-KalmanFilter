import numpy as np
import math
import matplotlib.pyplot as plt

# noise
Q = 1
R = 10
# time 
dt = 0.1
endtime = 20

def observation(xTrue, z):
    xTrue = motion_model(xTrue)
    z     = observation_model(xTrue)

    return xTrue, z

def motion_model(x):
    x = x + np.random.randn() * Q
    return x

def observation_model(x):
    z = x + np.random.randn() * R
    return z

def kalman_filter(xEst, PEst, z):
    # Predict
    xPred = motion_model(xEst)
    PPred = PEst + Q
    # Update
    K     = PPred / (PPred + R)
    zEst  = observation_model(xPred)
    xEst  = xPred + K * (z - zEst)
    PEst  = PPred - K * PPred

    return xEst, PEst

def main():
    # Initialize
    time  = 0.0
    z     = 0.0
    xEst  = 0.0
    xTrue = 0.0
    PEst  = 10.0
    # Data storage
    s_z     = z
    s_xEst  = xEst
    s_xTrue = xTrue
    s_PEst  = PEst
    s_time  = time

    while endtime >= time:
        time += dt
        
        xTrue, z   = observation(xTrue, z)
        xEst, PEst = kalman_filter(xEst, PEst, z)

        # save data
        s_z     = np.vstack((s_z, z))
        s_time  = np.vstack((s_time, time))
        s_xEst  = np.vstack((s_xEst, xEst))
        s_xTrue = np.vstack((s_xTrue, xTrue))
        s_PEst  = np.vstack((s_PEst, PEst))
    
    # Drow Graph
    plt.subplot(2, 1, 1)
    plt.plot(s_time[:, 0], s_z[:, 0], "-g", label = "Observation")
    plt.plot(s_time[:, 0], s_xTrue[:, 0], "-b", label = "True")
    plt.plot(s_time[:, 0], s_xEst[:, 0], "-r", label = "Estimate")
    plt.xlim([0, 20])
    plt.title("Random walk")
    plt.xlabel("Time[s]")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(s_time[:, 0], s_PEst[:, 0], "-c", label = "Nomal KF")
    plt.xlabel("Time[s]")
    plt.ylabel("Variance")
    plt.legend()
    plt.grid(True)

    plt.show()

if __name__ == '__main__':
    main()

