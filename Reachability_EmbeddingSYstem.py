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

# Define the initial set X0 = [-0.5, 0.5] x [-0.5, 0.5]
x1_range = np.linspace(-0.5, 0.5, 10)
x2_range = np.linspace(-0.5, 0.5, 10)
initial_conditions = [(x1, x2) for x1 in x1_range for x2 in x2_range]

# Run the simulation for each initial condition
trajectories = []

for (x1_0, x2_0) in initial_conditions:
    x1 = x1_0
    x2 = x2_0
    trajectory = [(x1, x2)]
    
    for _ in range(n_steps):
        x1, x2 = rk4_step(x1, x2, dt)
        trajectory.append((x1, x2))
    
    trajectories.append(trajectory)

# Plot the forward reachable set results
plt.figure(figsize=(8, 6))
for trajectory in trajectories:
    x1_traj, x2_traj = zip(*trajectory)
    plt.plot(x1_traj, x2_traj, 'g-', alpha=0.7)  # Forward reachable set in green

# Initial set
plt.scatter(x1_range, x2_range, color='orange', label='Initial Set X0')

# Now, compute and plot the over-approximation of the embedding system
# Extremes of the initial set for the over-approximation (corners of the initial condition space)
extreme_conditions = [(-0.5, -0.5), (-0.5, 0.5), (0.5, -0.5), (0.5, 0.5)]

# Run the simulation for each extreme condition
embedding_trajectories = []

for (x1_0, x2_0) in extreme_conditions:
    x1 = x1_0
    x2 = x2_0
    trajectory = [(x1, x2)]
    
    for _ in range(n_steps):
        x1, x2 = rk4_step(x1, x2, dt)
        trajectory.append((x1, x2))
    
    embedding_trajectories.append(trajectory)

# Find the final states (after t=1) for all extreme trajectories
final_points = [trajectory[-1] for trajectory in embedding_trajectories]

# Extract the max and min values of the final points to define the rectangle
x1_vals = [p[0] for p in final_points]
x2_vals = [p[1] for p in final_points]

x1_min, x1_max = min(x1_vals), max(x1_vals)
x2_min, x2_max = min(x2_vals), max(x2_vals)

# Plot the over-approximation rectangle
rectangle_x = [x1_min, x1_max, x1_max, x1_min, x1_min]
rectangle_y = [x2_min, x2_min, x2_max, x2_max, x2_min]
plt.plot(rectangle_x, rectangle_y, 'r--', label='Over-approximation Rectangle')

# Formatting the plot
plt.title("Forward Reachable Set (green) and Over-approximation (red) after t=1")
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.grid(True)
plt.legend()
plt.xlim([-1, 5])
plt.ylim([-1, 3])
plt.show()
