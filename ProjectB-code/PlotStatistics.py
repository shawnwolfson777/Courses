import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoLocator

from SimulationEnums import *
import RobotSimulationKalman

class PlotStatistics:
    def __init__(self, simulation_defs, tests):
        self.simulation_defs = simulation_defs
        self.fig = []
        self.main_cmap = plt.cm.tab10  # Use a perceptually distinct colormap
        self.axes_color = [0.8, 0.8, 0.8]
        self.axes = {}
        
        self.deltaX = 0.15  # Increase deltaX for more space
        self.deltaY = 0.55  # Increase deltaY for more space
        self.startX = 0.2
        self.startY = 0.05
        
        self.tests = tests
        self.num_tests = len(self.tests)
        self.fig = [plt.figure(figsize=(17, 7)) for _ in range(self.num_tests)]
        
        self.mae_trajectory_values = []
        self.mae_drift_values = []
        self.message_counts = []
        self.mean_delay = []
        
        self.simulation()

        for test_num, test_name in enumerate(self.tests):
            # Use a single figure with subplots
            if test_num == 0:  # Create the shared figure on the first test
                self.fig = plt.figure(figsize=(17, 7))
                
            ax = self.fig.add_subplot(1, len(self.tests), test_num + 1, facecolor=self.axes_color)
            ax.set_title(f"{test_name}, protocol: {self.simulation_defs.protocol}", color='Black')
            ax.set_xticks([])
            ax.set_yticks([])
            self.axes[test_name] = ax

            if test_name == 'MAE-trajectory':
                ax.set_title("Mean estimated route offset from the ground truth, as a function of time")
                self.plot_mae_trajectory_comparison()
            elif test_name == 'MAE-drift':
                ax.set_title("Mean estimated offset from estimation to ground truth, as a function of time")
                self.plot_mae_drift_comparison()

    def simulation(self):
        # According to external simulation defs, makes multiple simulation and extracts statistics for plotting
        for delayed_kalman in self.simulation_defs.delayed_kalman:
            for delay_window in self.simulation_defs.delay_windows:
                for no_erasure in self.simulation_defs.no_erasures_communication:
                    for idx, communication_error in enumerate(self.simulation_defs.communication_errors):
                        for type_idx, communication_type in enumerate(self.simulation_defs.communication_types):
                            self.robot_simulation = RobotSimulationKalman.Simulation(time_steps=self.simulation_defs.time_steps, dt=self.simulation_defs.dt, communication_error_base=communication_error, delay_type=communication_type, no_erasures_communication=no_erasure, delay_window=delay_window, delayed_kalman=delayed_kalman)
                            tot_mae_trajectory = np.zeros(self.robot_simulation.t-1)
                            tot_mae_drift = np.zeros(self.robot_simulation.t-1)
                            message_count = np.zeros(self.robot_simulation.t-1)
                            communication_delay = np.zeros(self.robot_simulation.t-1)
                            
                            for t in range(1,self.robot_simulation.t):
                                for i in range(self.robot_simulation.num_robots):
                                    for tTraj in range(0,t):
                                        tot_mae_trajectory[t-1] += (np.linalg.norm(self.robot_simulation.positions[i,:,tTraj] - self.robot_simulation.positions_estimation[i,:,tTraj,t]))
                                    tot_mae_drift[t-1] += (np.linalg.norm(self.robot_simulation.positions[i,:,t] - self.robot_simulation.positions_estimation[i,:,t,t]))
                                    message_count[t-1] += self.robot_simulation.num_messages_arrived[i,t]
                                    communication_delay[t-1] += np.sum(self.robot_simulation.communication_delay[i,:,t])

                            self.mae_trajectory_values.append((tot_mae_trajectory / self.robot_simulation.num_robots, delay_window, no_erasure, communication_error, communication_type, delayed_kalman))
                            self.mae_drift_values.append((tot_mae_drift / self.robot_simulation.num_robots, delay_window, no_erasure, communication_error, communication_type, delayed_kalman))                           
                            self.message_counts.append((communication_error, communication_type, message_count / self.robot_simulation.num_robots))
                            tmp_message_count = message_count
                            tmp_message_count[message_count==10] = -1 # for excluding self measurements. will be 0 after devision
                            self.mean_delay.append(communication_delay / (tmp_message_count-10)) # -10 in message_count is for excluding self measurement that doesn't have delay
                            
                        if no_erasure:
                            break

    def plot_mae_trajectory_comparison(self):
        colors = plt.cm.tab20(np.linspace(0, 1, ((True in self.simulation_defs.no_erasures_communication) + (False in self.simulation_defs.no_erasures_communication)*len(self.simulation_defs.communication_errors)) * len(self.simulation_defs.communication_types) * len(self.simulation_defs.delay_windows)))
        
        # Determine the global y-axis limits
        global_min = min([np.min(mae[0]) for mae in self.mae_trajectory_values])
        global_max = max([np.max(mae[0]) for mae in self.mae_trajectory_values])
        ylim_margin = (global_max - global_min) * 0.2  # Add 20% margin
        global_min -= ylim_margin
        global_max += ylim_margin

        ax = self.axes['MAE-trajectory']
        for idx, (tot_mae_trajectory, delay_window, no_erasure, communication_error, communication_type, delayed_kalman) in enumerate(self.mae_trajectory_values):
            delay_str = "with delay" if communication_type.name != "IDEAL" else "without delay"
            delay_window_str = f", allowed delay={delay_window*self.simulation_defs.dt} [s]" if communication_type.name != "IDEAL" else ""
            delay_compensation_str = f", with compensation" if delayed_kalman else f""
            compensation_addition_str = delay_compensation_str if communication_type.name != "IDEAL" else ""
            
            if no_erasure:
                label = f'No erasures, {delay_str}{delay_window_str}{compensation_addition_str}'
                delay_compensation_str = ""
            else:
                label = f'Erasures prob={communication_error*100}%, {delay_str}{delay_window_str}{compensation_addition_str}'
     
            ax.plot(tot_mae_trajectory, label=label, color=colors[idx])
            ax.xaxis.set_major_locator(AutoLocator())
            ax.yaxis.set_major_locator(AutoLocator())
            ax.grid(True)

        # Set the global y-limits
        ax.set_ylim(global_min, global_max)
        ax.legend()
        # Set axis labels
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Mean Absolute Error [m]")
        # Adjust x-ticks to represent time
        num_points = self.simulation_defs.time_steps
        x_ticks = np.arange(0, num_points) * self.simulation_defs.dt
        # Set a reduced number of x-ticks
        tick_interval = max(1, len(x_ticks) // 10)  # Show only 10 ticks (or fewer if there are fewer points)
        ax.set_xticks(np.arange(0, num_points, tick_interval))
        ax.set_xticklabels([f"{x_ticks[i]:.1f}" for i in range(0, len(x_ticks), tick_interval)])
    
 
    def plot_mae_drift_comparison(self):
        colors = plt.cm.tab20(np.linspace(0, 1, ((True in self.simulation_defs.no_erasures_communication) + (False in self.simulation_defs.no_erasures_communication)*len(self.simulation_defs.communication_errors)) * len(self.simulation_defs.communication_types) * len(self.simulation_defs.delay_windows)))
        
        # Determine the global y-axis limits
        global_min = min([np.min(mae[0]) for mae in self.mae_drift_values])
        global_max = max([np.max(mae[0]) for mae in self.mae_drift_values])
        ylim_margin = (global_max - global_min) * 0.2  # Add 20% margin
        global_min -= ylim_margin
        global_max += ylim_margin

        ax = self.axes['MAE-drift']
        for idx, (tot_mae, delay_window, no_erasure, communication_error, communication_type, delayed_kalman) in enumerate(self.mae_drift_values):
            delay_str = "with delay" if communication_type.name != "IDEAL" else "without delay"
            delay_window_str = f", allowed delay={delay_window*self.simulation_defs.dt} [s]" if communication_type.name != "IDEAL" else ""
            delay_compensation_str = f", with compensation" if delayed_kalman else f""
            compensation_addition_str = delay_compensation_str if communication_type.name != "IDEAL" else ""
            
            if no_erasure:
                label = f'No erasures, {delay_str}{delay_window_str}{compensation_addition_str}'
                delay_compensation_str = ""
            else:
                label = f'Erasures prob={communication_error*100}%, {delay_str}{delay_window_str}{compensation_addition_str}'
     
            ax.plot(tot_mae, label=label, color=colors[idx])
            ax.xaxis.set_major_locator(AutoLocator())
            ax.yaxis.set_major_locator(AutoLocator())
            ax.grid(True)

        # Set the global y-limits
        ax.set_ylim(global_min, global_max)
        ax.legend()
        # Set axis labels
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Mean Absolute Error [m]")
        # Adjust x-ticks to represent time
        num_points = self.simulation_defs.time_steps
        x_ticks = np.arange(0, num_points) * self.simulation_defs.dt
        # Set a reduced number of x-ticks
        tick_interval = max(1, len(x_ticks) // 10)  # Show only 10 ticks (or fewer if there are fewer points)
        ax.set_xticks(np.arange(0, num_points, tick_interval))
        ax.set_xticklabels([f"{x_ticks[i]:.1f}" for i in range(0, len(x_ticks), tick_interval)])
               
    def plot_message_count_comparison(self):
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.simulation_defs.communication_errors) * len(self.simulation_defs.communication_types)))
        
        # Determine the global y-axis limits
        global_min = min([np.min(count[2]) for count in self.message_counts])
        global_max = max([np.max(count[2]) for count in self.message_counts])
        ylim_margin = (global_max - global_min) * 0.2  # Add 20% margin
        global_min -= ylim_margin
        global_max += ylim_margin

        ax = self.axes['Message Count']
        for idx, (communication_error, communication_type, message_count) in enumerate(self.message_counts):
            label = f'Error Base={communication_error*100}%, Type={communication_type.name}'
            ax.plot(message_count, label=label, color=colors[idx])
            ax.xaxis.set_major_locator(AutoLocator())
            ax.yaxis.set_major_locator(AutoLocator())
            ax.grid(True)

        # Set the global y-limits
        ax.set_ylim(global_min, global_max)
        ax.legend()
        
    def plot_mean_delay(self):
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.simulation_defs.communication_errors) * len(self.simulation_defs.communication_types)))
        
        # Determine the global y-axis limits
        global_min = min([np.min(delay) for delay in self.mean_delay])
        global_max = max([np.max(delay) for delay in self.mean_delay])
        ylim_margin = (global_max - global_min) * 0.2  # Add 20% margin
        global_min -= ylim_margin
        global_max += ylim_margin

        ax = self.axes['Mean Delay']
        for idx, (mean_delay) in enumerate(self.mean_delay):
            label = f'mean delay'
            ax.plot(mean_delay, label=label, color=colors[idx])
            ax.xaxis.set_major_locator(AutoLocator())
            ax.yaxis.set_major_locator(AutoLocator())
            ax.grid(True)

        # Set the global y-limits
        ax.set_ylim(global_min, global_max)
        ax.legend()

class SimulationDefs:
    def __init__(self, sim_type):
        if sim_type == "ideal world":
            self.no_erasures_communication = [True]
            self.communication_errors = [0]
            self.communication_types=[Delay.IDEAL]
            self.protocol='UDP'
            self.delay_windows = [0]
            self.dt = 0.1 
            self.time_steps = 2001
            self.delayed_kalman = [False]
 
        if sim_type == "ideal communication vs no communication":
            self.no_erasures_communication = [False]
            self.communication_errors = [0, 1]
            self.communication_types = [Delay.IDEAL]
            self.protocol='UDP'
            self.delay_windows = [0]
            self.dt = 0.1 
            self.time_steps = 2001
            self.delayed_kalman = [False]
                       
        if sim_type == "different erasures probs":
            self.no_erasures_communication = [True, False]
            self.communication_errors = [0.2, 0.4, 0.6, 0.8, 1]
            self.communication_types=[Delay.IDEAL]
            self.protocol='UDP'
            self.delay_windows = [0]
            self.dt = 0.1  
            self.time_steps = 2001
            self.delayed_kalman = [False]
            
        if sim_type == "delay vs no delay":
            self.no_erasures_communication = [True]
            self.communication_errors = [0]
            self.communication_types=[Delay.GAUSSIAN, Delay.IDEAL]
            self.protocol='UDP'
            self.delay_windows = [np.inf]
            self.dt = 0.1
            self.time_steps = 2001
            self.delayed_kalman = [False]
               
        if sim_type == "window for delay and delay compensation":
            self.no_erasures_communication = [True]
            self.communication_errors = [0]
            self.communication_types=[Delay.GAUSSIAN]
            self.protocol='UDP'
            self.delay_windows = [0, 10, 20, 30, 40, 50]
            self.dt = 0.1
            self.time_steps = 2001
            self.delayed_kalman = [True]
            
        if sim_type == "just window for delay":
            self.no_erasures_communication = [True]
            self.communication_errors = [0]
            self.communication_types=[Delay.GAUSSIAN]
            self.protocol='UDP'
            self.delay_windows = [0, 10, 20, 30, 40, 50]
            self.dt = 0.1
            self.time_steps = 2001
            self.delayed_kalman = [False]
            
        if sim_type == "mean delay":
            self.no_erasures_communication = [True]
            self.communication_errors = [0]
            self.communication_types=[Delay.GAUSSIAN]
            self.protocol='UDP'
            self.delay_windows = [np.inf]
            self.dt = 0.1
            self.time_steps = 2001
            self.delayed_kalman = [False]

        if sim_type == "All together":
            self.no_erasures_communication = [False]
            self.communication_errors = [0, 0.2, 0.4, 0.6, 0.8, 1]
            self.communication_types=[Delay.GAUSSIAN]
            self.protocol='UDP'
            self.delay_windows = [20, 30]
            self.dt = 0.1
            self.time_steps = 2001
            self.delayed_kalman = [True]            
        
if __name__ == "__main__":
    
    #=====================================================================
    # This main runs all the tests used in our report.
    #=====================================================================
    sim_type_1 = "ideal world" # First statistics of the paper
    sim_type_2 = "different erasures probs" # second statistics of the paper
    sim_type_3 = "delay vs no delay" # third statistics of the paper
    sim_type_4 = "just window for delay" # fourth statistics of the paper
    sim_type_5 = "window for delay and delay compensation" # fifth statistics of the paper
    sim_type_6 = "All together" # sixth statistics of the paper

    #=====================================================================
    # USER - enter here simulation type from the above options
    #=====================================================================
    defs = SimulationDefs(sim_type_5)
    #=====================================================================
    
    #=====================================================================
    # USER - Choose tests
    #=====================================================================
    # PlotStatistics(defs, tests=['MAE-drift', 'MAE-trajectory'])
    PlotStatistics(defs, tests=['MAE-trajectory'])
    # PlotStatistics(defs, tests=['Mean Delay'])
    #=====================================================================
    
    plt.show()  # Keep the plots open
