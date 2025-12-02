import numpy as np

class TrackData:

    def __init__(self, n_drones, time_step, simulation_time):
        self.n_drones = n_drones
        self.time_step = time_step
        self.N_init = int(1.1*np.ceil(simulation_time/time_step))
        self.counter = 0
        self.t_prev = 0
        self.t = np.zeros(self.N_init)
        self.state_full = np.zeros((n_drones, 13, self.N_init))
        self.centroid = np.zeros((3, self.N_init))
        self.scale = np.zeros(self.N_init)
        self.scale_ref = np.zeros(self.N_init)
        self.operation = "take_off"
        self.moment_operation = [0]
        self.e_ij_norms = np.zeros(((n_drones * (n_drones - 1) // 2), self.N_init))
        self.t = np.zeros(self.N_init)
        

    def track_waypoints(self, waypoints):
        self.waypoints = waypoints

    def track_scale_ref_norm(self, X_ref):
        centroid_init = np.mean(X_ref, axis=0) 
        sq_dist = np.sum((X_ref - centroid_init)**2, axis=1)                  
        self.scale_ref_norm = np.sqrt(np.mean(sq_dist))

    def track_state_full(self, t, operation, state_full, e_bearing, scale_ref):

        if operation != self.operation:
            self.moment_operation.append(self.counter)
            self.operation = operation

        if self.t_prev + self.time_step > t:
            return
        
        self.t[self.counter] = t
        # state_full[i, :, k] = [pos(3), quat(4), vel(6)] 
        self.state_full[:, :, self.counter] = state_full
        self.scale_ref[self.counter] = scale_ref
        pos_full = self.state_full[:, :3, self.counter]
        self.cal_maneuvers_formation(pos_full)
        self.cal_bearing_error(e_bearing)

        self.counter += 1
        self.t_prev = t

    def cal_bearing_error(self, e_bearing):
        k = 0
        for i in range(self.n_drones):
            for j in range(i + 1, self.n_drones):         
                e_ij = e_bearing[i, j, :]     
                norm_ij = np.linalg.norm(e_ij)
                self.e_ij_norms[k, self.counter] = norm_ij
                k += 1

    def cal_maneuvers_formation(self, pos_full):
        self.centroid[:, self.counter] = np.mean(pos_full, axis=0)                     
        sq_dist = np.sum((pos_full - self.centroid[:, self.counter])**2, axis=1)                  
        self.scale[self.counter] = np.sqrt(np.mean(sq_dist))/self.scale_ref_norm 

    def remove_unused_space(self):
        self.moment_operation.append(self.counter-1)
        self.t = self.t[:self.counter]
        self.state_full = self.state_full[:, :, :self.counter]
        self.scale_ref = self.scale_ref[:self.counter] 
        self.centroid = self.centroid[:, :self.counter]
        self.scale = self.scale[:self.counter]
        self.e_ij_norms = self.e_ij_norms[:, :self.counter]
        

if __name__ == "__main__":
    # simulation parameters
    n_drones = 5
    dt = 0.1
    T = 2.0

    tracker = TrackData(n_drones, dt, T)

    # simulate perfect unit-circle formation
    for k in range(tracker.N_init):

        state_full = np.zeros((n_drones, 13))

        # place drones on unit circle
        angles = np.linspace(0, 2*np.pi, n_drones, endpoint=False)
        pos = np.zeros((n_drones, 3))
        pos[:, 0] = np.cos(angles)   # x
        pos[:, 1] = np.sin(angles)   # y
        pos[:, 2] = 0                # z fixed

        state_full[:, :3] = pos
        state_full[:, 3:7] = np.array([1, 0, 0, 0])         # identity quaternion
        state_full[:, 7:10] = 0.0                           # zero velocities

        tracker.track_state_full(state_full)

    print("Centroid history:\n", tracker.centroid)
    print("\nScale history:\n", tracker.scale)