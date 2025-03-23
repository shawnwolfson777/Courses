import numpy as np
from SimulationEnums import *
from KalmanFilter import *
from KalmanWithDelay import *

# Simulation class:
# this class responsable for simulation robots movement in free limited space. The simulation includes update possition according to Ackermann model, position measurements by all the robots, and communication modeling.  

class Simulation:
    def __init__(self, seed=1, num_robots=10, time_steps=1000, dt=0.1, space_limit=100, border_threshold=18, max_acc=8, max_vel=10, min_vel=0, max_steering_angle=30, run_away_steering_angle=40,
                 wheelbase=2.5, damping_factor=0.8, collision_distance=20, range_meas_mean=0, internal_noise_std=5, self_meas_std=5, update_time_const=8, reset_delta_time_const=3, robot_length=4, robot_width=2,
                 wheel_radius=0.5, no_erasures_communication=True, communication_error_base=0.5, delay_type = Delay.IDEAL, delay_mean=25, delay_std=15, delay_window = 10, k1=0.01, k2=0, delayed_kalman=False):
        
        # Initial states
        self.t = 1
        self.finished_cycle = False

        np.random.seed(seed)

        # Define generic simulation parameters
        self.num_robots = num_robots
        self.time_steps = time_steps
        self.dt = dt  # time step duration
        self.space_limit = space_limit
        self.border_threshold = border_threshold  # Threshold to check if a robot is near the border
        
        # Define robots movement parameters
        self.max_acc = max_acc  # Maximum allowed acceleration magnitude
        self.max_vel = max_vel  # Maximum allowed velocity magnitude
        self.min_vel = min_vel # Minimum allowed velocity magnitude
        self.max_steering_angle = np.radians(max_steering_angle)  # Maximum allowed steering angle
        self.run_away_steering_angle = np.radians(run_away_steering_angle)  # Runaway steering angle
        self.wheelbase = wheelbase  # Distance between front and rear axles
        self.damping_factor = damping_factor  # Factor to slow down velocity when maximum acceleration is reached
        self.collision_distance = collision_distance  # Minimum distance to avoid collisions
        self.update_time_const = update_time_const # how many cycles until update steering angle and acceleration
        self.reset_delta_time_const = reset_delta_time_const # how many cycles pass until reset steering angle after update it
        
        # Define measurements parameters
        self.range_meas_mean = range_meas_mean # mean for range measurements
        self.internal_noise_std = internal_noise_std
        self.self_meas_std = self_meas_std  # GPS self-measurement noise standard deviation in meters

        # Define network parameters
        self.delay_type = delay_type
        self.delay_mean = delay_mean
        self.delay_std = delay_std
        self.delay_window = delay_window
        self.no_erasures_communication = no_erasures_communication
        self.communication_error_base = communication_error_base
        
        # Define filter parameters
        self.delayed_kalman = delayed_kalman
        
        # Robot dimensions (for plotting)
        self.robot_length = robot_length
        self.robot_width = robot_width
        self.wheel_radius = wheel_radius
        
        # Global structs
        self.adjusted_robots = np.zeros(self.num_robots)
        self.measurement_lists = [[] for _ in range(self.num_robots)]

        # Statistics structs
        self.num_messages_arrived = np.zeros((self.num_robots, self.time_steps)) # For every robot and for every time step, counts the number of arrived messages.
        self.communication_per_time_slot = np.zeros((self.num_robots, self.num_robots, self.time_steps), dtype=bool) # boolean parameter - for every time step of the simulation, indicates if robot "a" got a message from robot "b".
        self.communication_delay = np.zeros((self.num_robots, self.num_robots, self.time_steps)) # For every robot holds the sum(!) delay of the messages arrived from robot i in time t
        self.single_msg_communication_delay = np.zeros((self.num_robots, self.num_robots, self.time_steps)) # For every robot holds the delay of the newest message arrived from robot i in time t. Used for the GUI to plot delayed messages

        # Initialize robots' properties
        self.positions = np.zeros((self.num_robots, 2, self.time_steps))
        self.orientations = np.zeros((self.num_robots, self.time_steps))  # Heading angles
        self.velocities = np.zeros((self.num_robots, self.time_steps))
        self.steering_angles = np.zeros((self.num_robots, self.time_steps))
        self.accelerations = np.zeros((self.num_robots, self.time_steps))
        self.positions_estimation = np.zeros((self.num_robots, 2, self.time_steps, self.time_steps)) # For every robot holds the estimated route in every time step.

        self.rand_steering_angle = np.random.uniform(low=-1, high=1, size=(self.num_robots, self.time_steps))
    
        # Initial positions, velocities, orientations, and steering angles
        for i in range(self.num_robots):
            self.positions[i, :, 0] = (np.random.rand(2) * 2 - 1) * self.space_limit # Initial position ~ U[-space_limit,space_limit]
            self.orientations[i, 0] = np.random.rand() * 2 * np.pi - np.pi # Initial orientation ~ U[-pi,pi]
            self.velocities[i, 0] = (np.random.rand()) * self.max_vel # Initial velocity ~ U[0, max_vel]
            self.steering_angles[i, 0] = 0
            self.accelerations[i, 0] = (np.random.rand() * self.max_acc) # Initial acceleration ~ U[0, max_acc]
            
        for i in range(self.num_robots):
            self.measurement_lists[i].append((i, 0, 0, self.positions[i, :, 0], self.positions[i, :, 0]))
        
        if self.delayed_kalman:
            self.kalman_filters = [
                KalmanFilterWithDelay(
                    robot_id = robot_id,
                    dt=dt,
                    wheelbase=wheelbase,
                    process_noise_std=np.array([self_meas_std, self_meas_std, 1*np.pi/180]),
                    initial_state_cov=np.array([1, 1, 1*np.pi/180]),
                    measurements_base_noise=internal_noise_std,
                    self_meas_std = self_meas_std,
                    space_limit = space_limit,
                    k1=k1,
                    k2=k2,
                    buffer_size = self.delay_window+1
                ) for robot_id in range(num_robots)
            ]
            
        else:
            self.kalman_filters = [
                KalmanFilter(
                    robot_id = robot_id,
                    dt=dt,
                    wheelbase=wheelbase,
                    process_noise_std=np.array([self_meas_std, self_meas_std, 1*np.pi/180]),
                    initial_state_cov=np.array([1, 1, 1*np.pi/180]),
                    measurements_base_noise=internal_noise_std,
                    self_meas_std = self_meas_std,
                    space_limit = space_limit,
                    k1=k1,
                    k2=k2
                ) for robot_id in range(num_robots)                
            ]
                
        self.run_simulation()

    

    def finish_cycle(self):
        self.adjusted_robots = np.zeros(self.num_robots)
        self.finished_cycle = True

    def start_cycle(self):
        self.finished_cycle = False


    def run_simulation(self):
        self.init_predictions()
        while self.t < self.time_steps:
            self.start_cycle()
            self.update_positions()
            self.predict_position()
            self.finish_cycle()
            self.t += 1

    def update_positions(self):
        for i in range(self.num_robots):
            # Extracting previous movement properties
            x, y = self.positions[i, :, self.t-1]
            theta = self.orientations[i, self.t-1]
            v = self.velocities[i, self.t-1]
            delta = self.steering_angles[i, self.t-1]
            a = self.accelerations[i, self.t-1]

            # calculate new acceleration and steering angle, every update_time_const time steps.
            if self.t % self.update_time_const == 0:
                # Update random acceleration addition
                random_acceleration = (np.random.rand() * 2 - 1) * 0.2 # Random acc addition ~ U[-0.2, 0.2] 
                self.accelerations[i, self.t] = np.clip(a + random_acceleration, -self.max_acc, self.max_acc)
                # Apply damping factor if the max acceleration is reached
                if np.abs(self.accelerations[i, self.t]) == self.max_acc:
                    self.accelerations[i, self.t] = -self.accelerations[i, self.t] * self.damping_factor        
                # Update steering angle randomly
                self.steering_angles[i, self.t] = self.rand_steering_angle[i, self.t] * self.max_steering_angle
            else: # keeping acceleration the same as previous time step
                self.accelerations[i, self.t] = self.accelerations[i, self.t-1]
                
                # Reset steering angle after update_time_const cycles, for allowing "clean" robots turns
                if self.t % self.update_time_const > self.reset_delta_time_const:
                    self.steering_angles[i, self.t] = 0
                else:
                    self.steering_angles[i, self.t] = delta
            
            self.steering_angles[i, self.t], self.accelerations[i,self.t] = check_direction_and_adjust([x, y], v, a, self.steering_angles[i, self.t], theta + self.dt * v * np.tan(delta) / self.wheelbase, self.dt, self.space_limit, self.border_threshold, self.run_away_steering_angle, i)
            
            # Update velocity with acceleration
            self.velocities[i, self.t] = np.clip(v + a*self.dt, self.min_vel, self.max_vel)

            # Check if car stopped:
            if self.velocities[i, self.t] == 0:
                self.accelerations[i, self.t] = (np.random.rand() + 1) # Initial acceleration ~ U[1, 2]

            # Ackermann model equations
            x_dot = v * np.cos(theta)
            y_dot = v * np.sin(theta)
            theta_dot = v * np.tan(delta) / self.wheelbase

            # Update positions and orientation
            self.positions[i, 0, self.t] = x + x_dot * self.dt
            self.positions[i, 1, self.t] = y + y_dot * self.dt
            self.orientations[i, self.t] = theta + theta_dot * self.dt

        # Simulate roboots measurements
        for measuring_robot in range(self.num_robots):
            for measured_robot in range(self.num_robots):
                transmission_position = self.positions[measuring_robot, :, self.t]
                curr_distance = np.linalg.norm(transmission_position - self.positions[measured_robot, :, self.t])
                if measured_robot == measuring_robot: # self measurement  
                    measurement_noise = self.self_meas_std 
                else:          
                    measurement_noise = self.internal_noise_std + (1/(np.sqrt(2)*self.space_limit)) * curr_distance # Noise increases with distance
                             
                measurement = self.positions[measured_robot, :, self.t] + np.random.normal(loc=self.range_meas_mean, scale=measurement_noise, size=2)

                if self.no_erasures_communication: # Ideal communication without erasures
                    if measuring_robot == measured_robot:
                        delay_cycles = 0 # for self measurement, we simulate message without delay
                    else:
                        delay_cycles = np.max([self.delay_std*np.random.randn()+self.delay_mean, 1]) # determines how many cycles it will take the message to arrive 
                    
                    # Update measurement_lists with new measurement. 
                    self.measurement_lists[measured_robot].append((measuring_robot, self.t, delay_cycles, transmission_position, measurement))
                elif np.random.rand() >= self.communication_error_base or measuring_robot == measured_robot:
                    if measuring_robot == measured_robot:
                        delay_cycles = 0 # for self measurement, we simulate message without delay
                    else:
                        delay_cycles = np.max([self.delay_std*np.random.randn()+self.delay_mean, 1]) # determines how many cycles it will take the message to arrive 
                    
                    # Update measurement_lists with new measurement.  
                    self.measurement_lists[measured_robot].append((measuring_robot, self.t, delay_cycles, transmission_position, measurement))
                                    
        self.detect_collisions()
            
        
    def init_predictions(self):
        # This function initiate the first position estimation of every robot.
        for i in range(self.num_robots):
            assert len(self.measurement_lists[i]) == 1
            _, _, _, _, meas = self.measurement_lists[i].pop(0)
            if self.delayed_kalman:
                self.kalman_filters[i].init_predictions((*meas,self.orientations[i,0]), [self.velocities[i, 0], self.steering_angles[i, 0]])
            else:
                self.kalman_filters[i].init_predictions((*meas,self.orientations[i,0])) 
            self.positions_estimation[i, :, 0, 0] = self.kalman_filters[i].state_est[:-1]
            self.num_messages_arrived[i, 0,] = 1
            self.communication_per_time_slot[i,i,0] = True
            
    def predict_position(self):
        # This function simulates position estimation of every robot, in a single time step.
        for i in range(self.num_robots):
            measurements = [] # Will hold the actualy recieved messages in this current cycle. 
            new_measurement_list = [] # Will hold the remaining messages that weren't recived yet due to delay.
            
            while self.measurement_lists[i]:
                measuring_robot, transmit_time, relative_arrival_cycles, trasnmit_pos, meas = self.measurement_lists[i][0] # Extract a single measurement data.
                delay_condition_met = False
                if self.t - transmit_time > self.delay_window: # throwing away old
                    self.measurement_lists[i].pop(0)
                    continue
                
                if self.delay_type == Delay.IDEAL:
                    delay_condition_met = True
                elif self.delay_type == Delay.GAUSSIAN:
                    delay_in_cycles = self.t - transmit_time
                    if delay_in_cycles >= relative_arrival_cycles or measuring_robot == i: # the message still didn't arrive
                        delay_condition_met = True

                if delay_condition_met: # if this single message should get to the robot, we insert it to measurements list     
                    self.measurement_lists[i].pop(0)
                    measurements.append((measuring_robot, transmit_time, relative_arrival_cycles, trasnmit_pos, meas))
                    self.num_messages_arrived[i, self.t] += 1
                    self.communication_per_time_slot[i,measuring_robot,self.t] = True
                    self.communication_delay[i,measuring_robot,self.t] += (self.t - transmit_time)
                    self.single_msg_communication_delay[i,measuring_robot,self.t] = (self.t - transmit_time)
                else: # the message should not be account for position estimation
                    new_measurement_list.append((measuring_robot, transmit_time, relative_arrival_cycles, trasnmit_pos, meas))
                    self.measurement_lists[i].pop(0)   
            
            self.positions_estimation[i, :, :, self.t] = self.positions_estimation[i, :, :, self.t-1] # copy the estimated route from previous time step.  

            # Predict the next state
            control_input = [self.velocities[i, self.t-1], self.steering_angles[i, self.t-1]]
            
            if self.delayed_kalman:
                self.kalman_filters[i].kalman_wrapper(measurements, current_time=self.t, control=control_input)
                for iUpdate in range(min(self.t,self.delay_window+1)): 
                    state_est = self.kalman_filters[i].state_buffer[-(iUpdate+1)]
                    self.positions_estimation[i, :, self.t-iUpdate, self.t] = state_est[:-1]
            else:
                self.kalman_filters[i].predict(control_input)
                if measurements:
                    self.kalman_filters[i].update(measurements)
                self.positions_estimation[i, :, self.t, self.t] = self.kalman_filters[i].state_est[:-1]
            
            self.measurement_lists[i] = new_measurement_list
            
    def detect_collisions(self):
        for i in range(self.num_robots):
            for j in range(i + 1, self.num_robots):
                prev_dist = np.linalg.norm(self.positions[i, :, self.t-1] - self.positions[j, :, self.t-1])
                curr_dist = np.linalg.norm(self.positions[i, :, self.t] - self.positions[j, :, self.t])
                if curr_dist < self.collision_distance and curr_dist < prev_dist and self.adjusted_robots[i] == False and self.adjusted_robots[j] == False:
                    # Update accelerations to avoid collision
                    self.accelerations[i, self.t] = self.max_acc / 2
                    self.accelerations[j, self.t] = self.max_acc / 2
                    # Update steering angles to avoid collision
                    self.adjusted_robots[i] = True
                    self.adjusted_robots[j] = True
                    future_orientation_max = self.orientations[i, self.t] + self.dt * self.velocities[i, self.t] * np.tan(self.max_steering_angle) / self.wheelbase
                    future_orientation_min = self.orientations[i, self.t] + self.dt * self.velocities[i, self.t] * np.tan(-self.max_steering_angle) / self.wheelbase
                    
                    future_x_i_max = self.positions[i, 0, self.t] + self.velocities[i, self.t] * np.cos(future_orientation_max)*self.dt
                    future_y_i_max = self.positions[i, 1, self.t] + self.velocities[i, self.t] * np.sin(future_orientation_max)*self.dt
                    future_x_j_min = self.positions[j, 0, self.t] + self.velocities[j, self.t] * np.cos(future_orientation_min)*self.dt
                    future_y_j_min = self.positions[j, 1, self.t] + self.velocities[j, self.t] * np.sin(future_orientation_min)*self.dt
                    
                    future_x_i_min = self.positions[i, 0, self.t] + self.velocities[i, self.t] * np.cos(future_orientation_min)*self.dt
                    future_y_i_min = self.positions[i, 1, self.t] + self.velocities[i, self.t] * np.sin(future_orientation_min)*self.dt
                    future_x_j_max = self.positions[j, 0, self.t] + self.velocities[j, self.t] * np.cos(future_orientation_max)*self.dt
                    future_y_j_max = self.positions[j, 1, self.t] + self.velocities[j, self.t] * np.sin(future_orientation_max)*self.dt
                    
                    future_dist_1 = np.linalg.norm([future_x_i_max - future_x_j_min, future_y_i_max -  future_y_j_min])
                    future_dist_2 = np.linalg.norm([future_x_i_min - future_x_j_max, future_y_i_min -  future_y_j_max])
                    
                
                    self.steering_angles[i, self.t] = self.max_steering_angle if future_dist_1 > future_dist_2 else -self.max_steering_angle
                    self.steering_angles[j, self.t] = -self.max_steering_angle if future_dist_1 <= future_dist_2 else self.max_steering_angle
                    
                
def check_direction_and_adjust(pos, vel, acc, steering_angle, orientation, dt, space_limit, border_threshold, max_steering_angle,i):
    new_delta = steering_angle
    new_acc = acc
    
    # Calculate future position based on current velocity and orientation
    future_x = pos[0] + vel * np.cos(orientation)*dt
    future_y = pos[1] + vel * np.sin(orientation)*dt

    # Check if near left or right border and adjust
    if future_x < -space_limit + border_threshold and np.cos(orientation) < 0:
        new_delta = -max_steering_angle if np.sin(orientation) > 0 else max_steering_angle  # Adjust steering to move away from border
    elif future_x > space_limit - border_threshold and np.cos(orientation) >= 0:
        new_delta = max_steering_angle if np.sin(orientation) >= 0 else -max_steering_angle  # Adjust steering to move away from border        
    # Check if near bottom or top border and adjust
    elif future_y < -space_limit + border_threshold and np.sin(orientation) < 0:
        new_delta = max_steering_angle if np.cos(orientation) > 0 else -max_steering_angle  # Adjust steering to move away from border
    elif future_y > space_limit - border_threshold and np.sin(orientation) >= 0:
        new_delta = -max_steering_angle if np.cos(orientation) > 0 else max_steering_angle  # Adjust steering to move away from border

    # Limit the steering angle to the maximum allowed value
    new_delta = np.clip(new_delta, -max_steering_angle, max_steering_angle)
    
    return new_delta, new_acc
