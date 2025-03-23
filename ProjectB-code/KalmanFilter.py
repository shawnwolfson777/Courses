import numpy as np

class KalmanFilter:
    def __init__(self, robot_id, dt, wheelbase, process_noise_std, initial_state_cov, measurements_base_noise, self_meas_std, space_limit, k1, k2):
        self.robot_id = robot_id
        self.dt = dt
        self.space_limit = space_limit
        self.wheelbase = wheelbase
        self.Q = np.diag(process_noise_std**2)
        self.measurements_base_noise = measurements_base_noise
        self.self_meas_std = self_meas_std
        self.k1 = k1
        self.k2 = k2
        self.P = np.diag(initial_state_cov**2)
        self.H = np.array([[1, 0, 0], [0, 1, 0]])
        self.state_est = None

    def init_predictions(self, measurements): # in first cycle we can't predict and just take the first measurement
        x, y, theta = measurements
        self.state_est = np.array([x, y, theta])
        
    def predict(self, control):
        x, y, theta = self.state_est
        v, delta = control
        
        A = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])
        
        B = np.array([
            [np.cos(theta) * self.dt, 0],
            [np.sin(theta) * self.dt, 0],
            [np.tan(delta)* self.dt / self.wheelbase ,0]
        ])
        
        self.state_est = A @ self.state_est + B @ control
        self.P = A @ self.P @ A.T + self.Q
        
    def update(self, measurements):
        for measuring_robot, transmit_time, relative_arrival_cycles, transmit_pos, measurement in measurements:
            distance = np.linalg.norm(transmit_pos - self.state_est[:2])
            if measuring_robot == self.robot_id:
                measurement_noise_std = self.self_meas_std
            else:
                measurement_noise_std = self.measurements_base_noise + (1/(np.sqrt(2)*self.space_limit)) * distance
            R = np.diag([(measurement_noise_std)**2,(measurement_noise_std)**2])

            K = self.P @ self.H.T @ np.linalg.inv(self.H @ self.P @ self.H.T + R)
            self.state_est += K @ (measurement - self.H @ self.state_est)
            self.P = (np.eye(3) - K @ self.H) @ self.P
            
