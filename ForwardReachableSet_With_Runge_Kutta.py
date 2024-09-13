import numpy as np
import matplotlib.pyplot as plt

# System dynamics
def f1(x1, x2):
    return x2**2 + 2

def f2(x1, x2):
    return x1

# Runge-Kutta 4th Order (RK4) for a single time step
def rk4_step(x1, x2, dt):
    # Compute k1
    k1_x1 = dt * f1(x1, x2)
    k1_x2 = dt * f2(x1, x2)
    
    # Compute k2
    k2_x1 = dt * f1(x1 + 0.5 * k1_x1, x2 + 0.5 * k1_x2)
    k2_x2 = dt * f2(x1 + 0.5 * k1_x1, x2 + 0.5 * k1_x2)
    
    # Compute k3
    k3_x1 = dt * f1(x1 + 0.5 * k2_x1, x2 + 0.5 * k2_x2)
    k3_x2 = dt * f2(x1 + 0.5 * k2_x1, x2 + 0.5 * k2_x2)
    
    # Compute k4
    k4_x1 = dt * f1(x1 + k3_x1, x2 + k3_x2)
    k4_x2 = dt * f2(x1 + k3_x1, x2 + k3_x2)
    
    # Update x1 and x2
    x1_next = x1 + (1/6) * (k1_x1 + 2*k2_x1 + 2*k3_x1 + k4_x1)
    x2_next = x2 + (1/6) * (k1_x2 + 2*k2_x2 + 2*k3_x2 + k4_x2)
    
    return x1_next, x2_next

# Simulation parameters
dt = 0.01  # Time step
T = 1.0    # Total time
n_steps = int(T / dt)  # Number of steps

# Initial condition
x1_0 = 0.2
x2_0 = -0.3

# Initialize the trajectory
trajectory = [(x1_0, x2_0)]
x1 = x1_0
x2 = x2_0

# Run the simulation for the initial condition
for _ in range(n_steps):
    x1, x2 = rk4_step(x1, x2, dt)
    trajectory.append((x1, x2))

# Plot the trajectory
plt.figure(figsize=(8, 6))
x1_traj, x2_traj = zip(*trajectory)
plt.plot(x1_traj, x2_traj, 'b-', label="Trajectory from (0.2, -0.3)", alpha=0.7)

# Mark the initial point
plt.scatter(x1_0, x2_0, color='orange', label='Initial Condition (0.2, -0.3)')

# Formatting the plot
plt.title("Trajectory for Initial Condition (0.2, -0.3)")
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.grid(True)
plt.legend()
plt.xlim([-1, 5])
plt.ylim([-1, 3])
plt.show()
