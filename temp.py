import numpy as np
import matplotlib.pyplot as plt

class Quadcopter:
    def __init__(self, name, position):
        self.name = name
        self.pos = np.array(position, dtype=float)
        self.vel = np.zeros(2)

    def move(self, v, dt=0.1):
        self.vel = v
        self.pos += v * dt

def leader_trajectory(t):
    # Moves forward along x, sinusoidal in y
    return np.array([0.5 * t, np.sin(0.2 * t)])

def control_follower(follower, leader_pos, offset, k_p=1.5):
    desired_pos = leader_pos + offset
    error = desired_pos - follower.pos
    return k_p * error

# Initialize agents
leader = Quadcopter("leader", [0, 0])
f1 = Quadcopter("follower1", [-2, -1])
f2 = Quadcopter("follower2", [-2, 1])

# Data for plotting
history = {'leader': [], 'f1': [], 'f2': []}

# Run simulation
for t in np.linspace(0, 20, 400):
    leader.pos = leader_trajectory(t)

    v1 = control_follower(f1, leader.pos, np.array([-2, -1]))
    v2 = control_follower(f2, leader.pos, np.array([-2,  1]))

    f1.move(v1)
    f2.move(v2)

    history['leader'].append(leader.pos.copy())
    history['f1'].append(f1.pos.copy())
    history['f2'].append(f2.pos.copy())

# Plot
plt.figure(figsize=(8, 6))
for name, traj in history.items():
    traj = np.array(traj)
    plt.plot(traj[:, 0], traj[:, 1], label=name)
plt.legend()
plt.title("Leader–Follower Formation Control")
plt.xlabel("x")
plt.ylabel("y")
plt.axis('equal')
plt.grid()
plt.show()
