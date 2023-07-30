from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
from scipy.linalg import block_diag
import numpy as np

def get_kalman_filter_config_from_data(data, dt = 2., starting_index = 0, R = 4.9):

    kalman_filter = KalmanFilter(dim_x=6, dim_z=3)# 6 state variables, 3 observed

    kalman_filter.F = np.array ([[1, dt, 0, 0, 0, 0], #kalman transition matrix
                      [0, 1, 0, 0, 0, 0],
                      [0, 0, 1, dt, 0, 0],
                      [0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 1, dt],
                      [0, 0, 0, 0, 0, 1]], dtype=float)

    kalman_filter.R *= R
    kalman_filter.Q *= .1

    kalman_filter.x = np.array([[data[starting_index][0], 0, data[starting_index][1], 0, data[starting_index][2], 0]], dtype=float).T #starting position, assuming it is correct
    kalman_filter.P = np.eye(6) * 500.
    kalman_filter.H = np.array([[1, 0, 0, 0, 0, 0], #how do we pass from measurement to position?
                     [0, 0, 1, 0, 0, 0], 
                     [0, 0, 0, 0, 1, 0]])
    
    q = Q_discrete_white_noise(dim=3, dt=dt, var=0.001) #white noise
    kalman_filter.Q = block_diag(q, q)

    return kalman_filter