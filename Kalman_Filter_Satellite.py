import numpy as np
import math
import matplotlib.pyplot as plt

# noise
Q = 0.64 ** -2
R = 1.0
# time 
dt = 1.0
endtime = 50

def observation(xTrue, z):
    xTrue, A, B = motion_model(xTrue)
    z, H        = observation_model(xTrue)

    return xTrue, z

def motion_model(x):
    A = np.matrix([[1.0, 1.0, 0.5, 0.5],
                   [0.0, 1.0, 1.0, 1.0],
                   [0.0, 0.0, 1.0, 0.0],
                   [0.0, 0.0, 0.0, 0.606]])
    
    B = np.matrix([0.0, 0.0, 0.0, 1.0]).T

    x = A * x + B * np.random.randn() * Q
    return x, A, B

def observation_model(x):
    H = np.matrix([1.0, 0.0, 0.0, 0.0])
    z = H * x + np.random.randn() * R
    return z, H

def kalman_filter(xEst, PEst, z):
    # Predict
    xPred, A, B = motion_model(xEst)
    PPred       = A * PEst * A.T + Q * B * B.T
    # Update
    zEst, H = observation_model(xPred)
    K       = PPred * H.T / (H * PPred * H.T + R)
    xEst    = xPred + K * (z - zEst)
    PEst    = (np.eye(len(xEst)) - K * H) * PPred

    return xEst, PEst

def main():
    # Initialize
    time  = 0.0
    z     = 1.25
    xEst  = np.matrix([1.25, 0.06, 0.01, -0.003]).T
    xTrue = np.matrix([1.25, 0.06, 0.01, -0.003]).T
    PEst  = np.diag([10.0, 10.0, 10.0, 10.0])
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
        s_xEst  = np.hstack((s_xEst, xEst))
        s_xTrue = np.hstack((s_xTrue, xTrue))
        s_PEst  = np.hstack((s_PEst, PEst))

    plt.subplot(2, 2, 1)
    plt.plot(s_time[:, 0], s_z[:, 0], "-g", label = "Observation")
    plt.plot(s_time[:, 0], np.array(s_xTrue[0, :]).flatten(), "-b", label = "True")
    plt.plot(s_time[:, 0], np.array(s_xEst[0, :]).flatten(), "-r", label = "Estimate")
    plt.xlabel("Time[s]")
    plt.ylabel("x1")
    plt.xlim([0, 50])
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(s_time[:, 0], np.array(s_xTrue[1, :]).flatten(), "-b", label = "True")
    plt.plot(s_time[:, 0], np.array(s_xEst[1, :]).flatten(), "-r", label = "Estimate")
    plt.xlabel("Time[s]")
    plt.ylabel("x2")
    plt.xlim([0, 50])
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(s_time[:, 0], np.array(s_xTrue[2, :]).flatten(), "-b", label = "True")
    plt.plot(s_time[:, 0], np.array(s_xEst[2, :]).flatten(), "-r", label = "Estimate")
    plt.xlabel("Time[s]")
    plt.ylabel("x3")
    plt.xlim([0, 50])
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(s_time[:, 0], np.array(s_xTrue[3, :]).flatten(), "-b", label = "True")
    plt.plot(s_time[:, 0], np.array(s_xEst[3, :]).flatten(), "-r", label = "Estimate")
    plt.xlabel("Time[s]")
    plt.ylabel("x4")
    plt.xlim([0, 50])
    plt.grid(True)
    plt.legend()

    plt.show()

if __name__ == '__main__':
    main()

