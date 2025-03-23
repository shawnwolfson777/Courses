# ProjectB
Project B repository.
The project simulates N robots moving freely (without crashing) in limited space, taking position measurements of every robot, and communicate with each other. The mission of every robot is to estimate its own position from the position measurements.
The user can choose between ideal communication without delay or gaussian distributed delay with some constant probabilty of packets erasure.

The project contains 6 files:
1. SimulationEnums.py - contains the available communication types
2. RobotSimulationKalman.py - the main code. Runs the simulaiton (update possitions, taking measurements, simulate communication and estimate positions).
3. KalmanFilter.py - Implement basic Kalman filter for position estimation 
4. KalmanWithDelay.py - Implement Kalman filter that can take into account messages that arrive in delay
5. PlotSimulation.py - Implement GUI for the simulation
6. PlotStatistics.py - Calculate and plot statistics of different simulation settings 