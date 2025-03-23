import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from matplotlib.widgets import Button
from matplotlib.patches import FancyArrowPatch

from SimulationEnums import *
import RobotSimulationKalman

class PlotSim():
    def __init__(self):
        # Init visualization states
        self.is_running = False
        self.show_line = False
        self.zoom = False
        self.show_predictions = False
        self.finished_sim = False
        self.button_clicked = False
        self.one_step = False
        self.prev_one_step = False
        self.communication = False
        self.curr_time = 0       
        # Run simulation
        self.robot_simulation = RobotSimulationKalman.Simulation(delay_type=Delay.GAUSSIAN, no_erasures_communication=True, dt=0.1, delay_window=50, delayed_kalman=True, time_steps=100)

        # Visual Parameters
        self.communication_lines = []
        self.communication_nodes = []
        self.subplot_communication_lines = []
        self.communication_nodes_radius = 2 
        self.tot_time_steps = self.robot_simulation.time_steps-1
        # Visualization & Colormaps
        self.deltaX = 0.12
        self.deltaY = 0.5
        self.subplot_x_start = 0.32
        self.subplot_y_start = 0.55
        self.subaxes_lim = 5
        self.fig_free_space = plt.figure(figsize=(17, 7))
        self.main_cmap = plt.cm.tab10  # Use a perceptually distinct colormap
        self.delay_colors = plt.cm.tab20(np.linspace(0, 1, self.robot_simulation.delay_window+1))
        self.axes_color = [0.1, 0.1, 0.1]
        self.axes_robot = []
        self.buttons = []
        self.main_ax_free_space = self.fig_free_space.add_axes([0.01, 0.05, 0.3, 0.9], facecolor=self.axes_color)
        self.main_ax_free_space.set_xlim(-self.robot_simulation.space_limit, self.robot_simulation.space_limit)
        self.main_ax_free_space.set_ylim(-self.robot_simulation.space_limit, self.robot_simulation.space_limit)
        self.main_ax_free_space.set_xticks([])
        self.main_ax_free_space.set_yticks([])
        self.main_ax_free_space.set_title(f"Time Step: {0} out of {self.tot_time_steps}")

        for i in range(self.robot_simulation.num_robots):
            ax = self.fig_free_space.add_axes([self.subplot_x_start + ((i % 5) * self.deltaX), self.subplot_y_start - self.deltaY * (i >= 5), 0.1, 0.4], facecolor=self.axes_color)
            ax.set_xlim(-self.robot_simulation.space_limit, self.robot_simulation.space_limit)
            ax.set_ylim(-self.robot_simulation.space_limit, self.robot_simulation.space_limit)
            ax.set_title(f"Robot {i + 1}", color='Black')
            ax.set_xticks([])
            ax.set_yticks([])
            self.axes_robot.append(ax)

            button_ax = self.fig_free_space.add_axes([self.subplot_x_start + ((i % 5) * self.deltaX), self.subplot_y_start - self.deltaY * (i >= 5) - 0.05, 0.1, 0.04], facecolor='lightgray')
            button = Button(button_ax, 'Expand')
            button.on_clicked(self.create_expand_function(i))
            self.buttons.append(button)

        # Initialize rectangles for robots
        self.robot_rectangles = []
        self.robot_wheels = []
        for i in range(self.robot_simulation.num_robots):
            rect = Rectangle((0, 0), self.robot_simulation.robot_length, self.robot_simulation.robot_width, linewidth=1, edgecolor=self.main_cmap(i), facecolor=self.main_cmap(i))
            self.main_ax_free_space.add_patch(rect)
            self.robot_rectangles.append(rect)
            wheels = []
            for _ in range(4):
                wheel = Circle((0, 0), self.robot_simulation.wheel_radius, linewidth=1, edgecolor='white', facecolor='black')  # Change to white color
                self.main_ax_free_space.add_patch(wheel)
                wheels.append(wheel)
            self.robot_wheels.append(wheels)

        # Pre-compute plots
        self.robot_plots = [self.main_ax_free_space.plot([], [], '*', linewidth=0.2, color=self.main_cmap(i))[0] for i in range(self.robot_simulation.num_robots)]
        self.robot_subplots = [self.axes_robot[i].plot([], [], linewidth=3, color=self.main_cmap(i))[0] for i in range(self.robot_simulation.num_robots)]
        self.robot_subplots_start = [self.axes_robot[i].plot([], [], '*', color=self.main_cmap(i))[0] for i in range(self.robot_simulation.num_robots)]
        self.robot_subplots_end = [self.axes_robot[i].plot([], [], 'o', color='yellow')[0] for i in range(self.robot_simulation.num_robots)]
        self.robot_prediction = [self.axes_robot[i].plot([], [], '-x', linewidth=1, color='white')[0] for i in range(self.robot_simulation.num_robots)]

        # Add buttons
        self.ax_start = plt.axes([0.925, 0.85, 0.05, 0.075])
        self.b_start = Button(self.ax_start, 'Start')
        self.b_start.on_clicked(self.start)

        self.ax_stop = plt.axes([0.925, 0.75, 0.05, 0.075])
        self.b_stop = Button(self.ax_stop, 'Stop')
        self.b_stop.on_clicked(self.stop)

        self.ax_toggle_line = plt.axes([0.925, 0.65, 0.05, 0.075])
        self.b_toggle_line = Button(self.ax_toggle_line, 'Main Line')
        self.b_toggle_line.on_clicked(self.toggle_line)

        self.ax_zoom = plt.axes([0.925, 0.55, 0.05, 0.075])
        self.b_zoom = Button(self.ax_zoom, 'Zoom In')
        self.b_zoom.on_clicked(self.zoom_supblots)

        self.ax_show_pred = plt.axes([0.925, 0.45, 0.05, 0.075])
        self.b_show_pred = Button(self.ax_show_pred, 'Show Pred')
        self.b_show_pred.on_clicked(self.show_pred)

        self.ax_time_step = plt.axes([0.925, 0.35, 0.05, 0.075])
        self.b_time_step = Button(self.ax_time_step, 'Next Step')
        self.b_time_step.on_clicked(self.time_step)
        
        self.ax_prev_time_step = plt.axes([0.925, 0.25, 0.05, 0.075])
        self.b_prev_time_step = Button(self.ax_prev_time_step, 'Prev Step')
        self.b_prev_time_step.on_clicked(self.prev_time_step)
        
        self.ax_communication_graph = plt.axes([0.925, 0.15, 0.05, 0.075])
        self.b_communication_graph = Button(self.ax_communication_graph, 'Network')
        self.b_communication_graph.on_clicked(self.show_communication)
        
        plt.show(block=False)

        self.update_plot_aux()

    def start(self, event):
        self.is_running = True
        if self.b_start.label._text == "Restart":
            self.finished_sim = True
        self.b_start.label.set_text("Continue")
        
    def stop(self, event):
        self.is_running = False

    def toggle_line(self, event):
        self.show_line = not self.show_line
        self.button_clicked = True

    def zoom_supblots(self, event):
        if not self.zoom:
            self.b_zoom.label.set_text("Zoom Out")
        else:
            self.b_zoom.label.set_text("Zoom In")
        self.zoom = not self.zoom
        self.button_clicked = True

    def show_pred(self, event):
        if not self.show_predictions:
            self.b_show_pred.label.set_text("Hide Pred")
        else:
            self.b_show_pred.label.set_text("Show Pred")
        self.button_clicked = True
        self.show_predictions = not self.show_predictions

    def time_step(self, even):
        if self.is_running:
            return
        else:
            self.one_step = True
    
    def prev_time_step(self, even):
        if self.is_running:
            return
        else:
            self.prev_one_step = True

    def show_communication(self, event):
        self.communication = not self.communication
        self.button_clicked = True
        
    def create_expand_function(self, i):
        def expand(event):
            fig, ax = plt.subplots()
            ax.plot(self.robot_simulation.positions[i, 0, :self.curr_time+1], self.robot_simulation.positions[i, 1, :self.curr_time+1], linewidth=3, color=self.main_cmap(i))
            ax.plot(self.robot_simulation.positions[i, 0, 0], self.robot_simulation.positions[i, 1, 0], 'o', color='yellow')
            ax.plot(self.robot_simulation.positions[i, 0, self.curr_time], self.robot_simulation.positions[i, 1, self.curr_time], 'x', color='red')
            ax.plot(self.robot_simulation.positions_estimation[i, 0, :self.curr_time+1, self.curr_time], self.robot_simulation.positions_estimation[i, 1, :self.curr_time+1, self.curr_time], '-x', linewidth=1, color='white')
            ax.set_xlim(-self.robot_simulation.space_limit, self.robot_simulation.space_limit)
            ax.set_ylim(-self.robot_simulation.space_limit, self.robot_simulation.space_limit)
            ax.set_title(f"Robot {i + 1} Expanded", color='black')  # Title color
            ax.set_facecolor(self.axes_color)  # Set axis background color
            plt.show()
        return expand
    
    def update_plot_aux(self):
        while True:
            self.update_plot()
            self.finished_sim = False
        
    def update_plot(self):
        curr_time = 0
        while curr_time <= self.robot_simulation.t-1:
            if self.finished_sim:
                return
            if curr_time == self.robot_simulation.t-1:
                self.b_start.label.set_text("Restart")
                self.is_running = False
                plt.pause(0.01)
            
            self.curr_time = curr_time
                
            while not self.is_running and curr_time > 0 and not self.button_clicked and not self.one_step and not self.prev_one_step:
                plt.pause(0.005)

            if not self.is_running and (self.button_clicked or curr_time == 0) and not self.one_step and not self.prev_one_step:
                self.button_clicked = False
            else:
                if self.one_step:
                    self.one_step = False
                if self.prev_one_step:
                    self.prev_one_step = False
                    curr_time = max(curr_time-1,0)
                else:
                    curr_time = min(curr_time+1,self.tot_time_steps)
            self.main_ax_free_space.set_title(f"Time Step: {curr_time} out of {self.tot_time_steps}")
            
            for i in range(self.robot_simulation.num_robots):
                if self.show_line:
                    self.robot_plots[i].set_data(self.robot_simulation.positions[i, 0, 0:curr_time+1], self.robot_simulation.positions[i, 1, 0:curr_time+1])
                else:
                    self.robot_plots[i].set_data([],[])

                
                self.robot_subplots[i].set_data(self.robot_simulation.positions[i, 0, 0:curr_time+1], self.robot_simulation.positions[i, 1, 0:curr_time+1])
                if self.zoom:
                    self.axes_robot[i].set_xlim(self.robot_simulation.positions[i, 0, curr_time]-self.subaxes_lim, self.robot_simulation.positions[i, 0, curr_time]+self.subaxes_lim)
                    self.axes_robot[i].set_ylim(self.robot_simulation.positions[i, 1, curr_time]-self.subaxes_lim, self.robot_simulation.positions[i, 1, curr_time]+self.subaxes_lim)
                else:
                    self.axes_robot[i].set_xlim(-self.robot_simulation.space_limit, self.robot_simulation.space_limit)
                    self.axes_robot[i].set_ylim(-self.robot_simulation.space_limit, self.robot_simulation.space_limit)
                
                self.robot_subplots_start[i].set_data(self.robot_simulation.positions[i, 0, 0], self.robot_simulation.positions[i, 1, 0])
                self.robot_subplots_end[i].set_data(self.robot_simulation.positions[i, 0, curr_time], self.robot_simulation.positions[i, 1, curr_time])
            
                if self.show_predictions:
                    self.robot_prediction[i].set_data(self.robot_simulation.positions_estimation[i, 0, :curr_time+1, curr_time], self.robot_simulation.positions_estimation[i, 1, :curr_time+1, curr_time])
                else:
                    self.robot_prediction[i].set_data([],[])

                # Update rectangles' positions and orientations
                x = self.robot_simulation.positions[i, 0, curr_time]
                y = self.robot_simulation.positions[i, 1, curr_time]
                theta = self.robot_simulation.orientations[i, curr_time]
                
                # Calculate the bottom-left corner of the rectangle
                rect_x = x + (self.robot_simulation.robot_width / 2) * np.sin(theta)
                rect_y = y - (self.robot_simulation.robot_width / 2) * np.cos(theta)

                self.robot_rectangles[i].set_xy((rect_x, rect_y))
                self.robot_rectangles[i].angle = np.degrees(theta)
                
                # Update wheel positions
                wheel_positions = [
                    (rect_x, rect_y),  # Rear-left
                    (rect_x - (self.robot_simulation.robot_width) * np.sin(theta), rect_y + (self.robot_simulation.robot_width) * np.cos(theta)),  # Rear-right
                    (rect_x + (self.robot_simulation.robot_length) * np.cos(theta), rect_y + (self.robot_simulation.robot_length) * np.sin(theta)),  # Front-left
                    (rect_x - (self.robot_simulation.robot_width) * np.sin(theta) + (self.robot_simulation.robot_length) * np.cos(theta), rect_y + (self.robot_simulation.robot_width) * np.cos(theta) + (self.robot_simulation.robot_length) * np.sin(theta))   # Front-right
                ]
                for wheel, (wheel_x, wheel_y) in zip(self.robot_wheels[i], wheel_positions):
                    wheel.set_center((wheel_x, wheel_y))
                    
            # Plot communication network
            if self.communication:
                self.plot_communication_lines(curr_time)
            else:
                self.clear_communication_lines()
                
            plt.draw()
            plt.pause(0.005)
                
    def plot_communication_lines(self, curr_time):
        self.clear_communication_lines()
        for i in range(self.robot_simulation.num_robots):
            for j in range(self.robot_simulation.num_robots):
                if i != j and self.robot_simulation.communication_per_time_slot[i, j, curr_time]:
                    start_pos = self.robot_simulation.positions[j, :, curr_time]
                    end_pos = self.robot_simulation.positions[i, :, curr_time]
                    delay = self.robot_simulation.single_msg_communication_delay[i, j, curr_time]

                    # Choose color based on delay
                    color = self.delay_colors[int(delay)]

                    arrow_main = FancyArrowPatch(start_pos, end_pos, color=color, arrowstyle='->', mutation_scale=10)
                    self.main_ax_free_space.add_patch(arrow_main)
                    self.communication_lines.append(arrow_main)
                    
                    
                    subplot_line, = self.axes_robot[i].plot([start_pos[0], end_pos[0]], [start_pos[1], end_pos[1]], color=color, linestyle='--', linewidth=1)
                    self.subplot_communication_lines.append(subplot_line)                    # arrow_sub_robot_axes = FancyArrowPatch(start_pos, end_pos, color=color, arrowstyle='-', mutation_scale=10)
                    communication_node = Circle((self.robot_simulation.positions[j, 0, curr_time], self.robot_simulation.positions[j, 1, curr_time]), self.communication_nodes_radius, linewidth=1, edgecolor=self.main_cmap(j), facecolor=self.main_cmap(j))
                    self.axes_robot[i].add_patch(communication_node)
                    self.communication_nodes.append(communication_node)

        # Add legend for delay colors
        legend_elements = [Rectangle((0, 0), 1, 1, color=self.delay_colors[i]) for i in range(self.robot_simulation.delay_window+1)]
        legend_labels = [f"Delay {i}" for i in range(self.robot_simulation.delay_window+1)]
        self.main_ax_free_space.legend(legend_elements, legend_labels, loc='upper right', title="Communication Delay")

                    
    def clear_communication_lines(self):
        for line in self.communication_lines:
            line.remove()
        for node in self.communication_nodes:
            node.remove()
        for line in self.subplot_communication_lines:
            line.remove() 
        self.communication_lines = []
        self.communication_nodes = []
        self.subplot_communication_lines = []

        self.main_ax_free_space.legend(handles=[], handlelength=0)

        
        
if __name__ == "__main__":
    PlotSim()