import numpy as np

# This class holds klaman filter that can handle with delay. The filter saves "buffer_size" (external input) of matrices and state vectors, and use them to iterative update the estimations 
class KalmanFilterWithDelay:
    def __init__(self, robot_id, dt, wheelbase, process_noise_std, initial_state_cov, measurements_base_noise, self_meas_std, space_limit, k1, k2, buffer_size):
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
        self.buffer_size = buffer_size # = delay window + 1
        self.state_buffer = []
        self.P_buffer = []
        self.control_buffer = []
        self.number_of_meas = []
        self.measurements_delay = [[] for _ in range(self.buffer_size)]


    def init_predictions(self, measurements, control): # in first cycle we can't predict and just take the first measurement
        x, y, theta = measurements
        self.state_est = np.array([x, y, theta])
        self.state_buffer.append(self.state_est.copy())
        self.P_buffer.append(self.P.copy())
        self.control_buffer.append(control.copy())
        self.number_of_meas.append(0)
        
    def predict(self, delay, control):
        state_index = -delay
        past_state = self.state_buffer[state_index-1]
        past_P = self.P_buffer[state_index-1]
        past_num_of_meas = self.number_of_meas[state_index-1]
        
        if delay == 0:
            curr_control = control
        else: 
            curr_control = self.control_buffer[state_index]
        
        x, y, theta = past_state
        v, delta = curr_control
        
        A = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])
        
        B = np.array([
            [np.cos(theta) * self.dt, 0],
            [np.sin(theta) * self.dt, 0],
            [np.tan(delta) * self.dt / self.wheelbase, 0]
        ])
        
        if state_index == 0: # predict new state and should add it to the buffer
            self.state_buffer.append(A @ past_state + B @ curr_control)
            self.P_buffer.append(A @ past_P @ A.T + self.Q)
            self.control_buffer.append(curr_control)
            self.number_of_meas.append(0)
        else: # fix previous prediction
            self.state_buffer[state_index] = A @ past_state + B @ curr_control
            self.P_buffer[state_index] = A @ past_P @ A.T + self.Q #/ (1+past_num_of_meas)
        

    # The following function wrapps the kalman prediction and filtering processes, and manage the input of delayed messages into the filter.      
    def kalman_wrapper(self, measurements, current_time, control):
        measurements_delay = self.measurements_delay
        
        for measuring_robot, transmit_time, relative_arrival_cycles, transmit_pos, measurement in measurements: # takes the data of a single measurement message, as defined in RobotSimulationKalman.py
            delay = current_time - transmit_time
            assert(delay < self.buffer_size)
            measurements_delay[delay].append((measuring_robot, transmit_time, relative_arrival_cycles, transmit_pos, measurement))

        first_delay_index = False
        for iDelay in range(self.buffer_size-1,-1,-1):
            # find the first delayed message
            if not measurements_delay[iDelay]:
                continue
            else:
                first_delay_index = iDelay
                break
                
        for iDelay in range(first_delay_index,-1,-1):
            self.predict(iDelay, control)
            self.update(measurements_delay[iDelay], iDelay)
            
        if len(self.state_buffer) > self.buffer_size: # pop the oldest state
            self.state_buffer.pop(0)
            self.P_buffer.pop(0)
        
        for iDelay in range(len(self.measurements_delay)-1,0,-1):
            self.measurements_delay[iDelay] = self.measurements_delay[iDelay-1]
        self.measurements_delay[0] = [] 

        self.state_est = self.state_buffer[-1]
        
    def update(self, measurements, delay):
        if delay == 0:
            state_index = -(delay+1)
        else:
            state_index = -(delay)
        curr_state = self.state_buffer[state_index].copy()
        curr_P = self.P_buffer[state_index].copy()
        for measuring_robot, transmit_time, relative_arrival_cycles, transmit_pos, measurement in measurements:            
            distance = np.linalg.norm(transmit_pos - curr_state[:2])
            
            if measuring_robot == self.robot_id:
                measurement_noise_std = self.self_meas_std
            else:
                measurement_noise_std = self.measurements_base_noise + (1/(np.sqrt(2)*self.space_limit)) * distance
                
            R = np.diag([(measurement_noise_std)**2,(measurement_noise_std)**2])

            K = curr_P @ self.H.T @ np.linalg.inv(self.H @ curr_P @ self.H.T + R)
            curr_state += K @ (measurement - self.H @ curr_state)
            curr_P = (np.eye(3) - K @ self.H) @ curr_P
            
        self.state_buffer[state_index] = curr_state
        self.P_buffer[state_index] = curr_P 
        self.number_of_meas[state_index] += len(measurements)

    def get_current_state(self):
        return self.state_est
