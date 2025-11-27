import numpy as np
from controls.quadcopter_controller import QuadcopterPIDController
from controls.consensus_controller import MultiAgentConsensus
from controls.pure_pursuit import PurePursuit
from path_planning.env import MapGridEnvironment3D
from path_planning.a_star_search import path_finding
from utilities.gen_multi_agent_graph import *

class ScenarioBearingbasedTrackingConsensus:

    def __init__(self):
        self.name = "Drones Formation using Bearing-based Consensus Algorithm"

    def init(self, sim, model, data):

        sim.cam.azimuth = 15
        sim.cam.elevation = -67
        sim.cam.distance =  32
        sim.cam.lookat =np.array([ 0.0 , 0.0 , 0.0 ])
        
        path, env = path_finding()
        self.env = env
        if path is None:
            path = sim.pos_ref

        self.path_tracking = PurePursuit(look_ahead_dist=1, waypoints=path)
        self.controllers = []
        self.formation_controller = MultiAgentConsensus(sim.num_drones, K=2, graph_type="complete")

        for _ in range(sim.num_drones):
            self.controllers.append(QuadcopterPIDController(sim.time_step))

        self.leader_controller = []
        for _ in range(2):
            self.leader_controller.append(QuadcopterPIDController(sim.time_step))

        self.tracking_flag = False
        self.formation_flag = False
        self.not_landing_flag = True
        self.altitude_ref = 5 * np.ones(sim.num_drones)
        self.t_prev = sim.data.time
        self.v_ref = np.zeros((sim.num_drones, 3))
        self.scale = 0.4

    def update(self, sim, model, data):
        n_drones = sim.num_drones
        dt = sim.data.time - self.t_prev
        print(sim.data.time)
        # --- Read current states ---
        X = np.zeros((n_drones, 3))
        dX = np.zeros((n_drones, 3))
        ddX = np.zeros((n_drones, 3))
        for i in range(n_drones):
            # print(i)
            quat = np.array(data.sensor(f'quat_{i}').data)
            X[i, :] = np.array(data.sensor(f'pos_{i}').data)
            dX[i, :] = np.array(data.sensor(f'vel_{i}').data)
            body_acc = np.array(data.sensor(f'acc_{i}').data) 
            ddX[i, :] = self.controllers[i].linear_acc_world(quat, body_acc)
            if abs(X[i, 2] - self.altitude_ref[i]) < 0.05:
                self.formation_flag = True
        # print(ddX[0, :])
        # --- Check if all drones reached altitude reference ---
        # self.tracking_flag = np.any(np.abs(X[:, 2] - self.altitude_ref) < 0.5) and 
        e_bearing = self.formation_controller.compute_bearing_error(X)

        if sim.data.time > 340 and self.not_landing_flag:
            self.not_landing_flag = False
            self.formation_controller.X_ref = gen_rectangle(sim.num_drones, spacing=0.8, agents_per_row=3)
            # X_ref = gen_sphere(self.n_agents, self.dim_state, radius=1)
            diff = self.formation_controller.X_ref[:, np.newaxis, :] - self.formation_controller.X_ref[np.newaxis, :, :]
            norm = np.linalg.norm(diff, axis=2, keepdims=True)
            norm = np.where(norm == 0, 1e-9, norm)   # avoid divide-by-zero
            self.formation_controller.dir_ref = -diff / norm
            self.altitude_ref = X[:, 2]
            self.scale = 1

        # print(f"time: {sim.data.time:.2f}, e_bearing: {np.linalg.norm(e_bearing):.2f}")
        if np.linalg.norm(e_bearing) < 10:
            self.tracking_flag = True

        if (not self.not_landing_flag) and (np.linalg.norm(e_bearing) < 1):
            self.altitude_ref = self.controllers[0].low_pass_filter(np.zeros(sim.num_drones), self.altitude_ref, alpha=0.5)


            # self.tracking_flag = False
        # print(np.linalg.norm(e_bearing))
        # --- Update camera once per timestep ---
        sim.cam.lookat = X.mean(axis=0)
        # sim.cam.azimuth += 0.1

        v_rep = self.env.compute_repulsive_velocity_multi(X, influence_distance=1, eta=0.01)

        # --- Compute formation control ---
        control_consensus = np.zeros_like(X)
        if self.tracking_flag and self.not_landing_flag:
            control_consensus = self.formation_controller.consensus_leader_vel_varying_law(
                X, dX, ddX, kp=100, kv=200
            )  # returns acceleration commands
            # --- Integrate reference velocity and altitude ---
            # self.v_ref += control_consensus*dt
            self.v_ref = self.controllers[0].low_pass_filter(self.v_ref + v_rep + control_consensus*dt, self.v_ref, alpha=0.1)      
            self.altitude_ref = self.controllers[0].low_pass_filter(self.altitude_ref + self.v_ref[:, 2] * dt, self.altitude_ref, alpha=0.5)
        elif self.formation_flag:
            control_consensus = self.formation_controller.consensus_law(X)
            self.v_ref = control_consensus + v_rep       
            self.altitude_ref += self.v_ref[:, 2] * dt


        # --- Compute individual drone control ---
        # print(np.array(data.sensor(f'pos_{0}').data))
        for i in range(n_drones):
            pos = np.array(data.sensor(f'pos_{i}').data)
            quat = np.array(data.sensor(f'quat_{i}').data)
            linvel = np.array(data.sensor(f'vel_{i}').data)
            angvel = np.array(data.sensor(f'gyro_{i}').data)
            vel = np.hstack((linvel, angvel))
            state = np.concatenate([pos, quat, vel])

            # Compute velocity-based control
            
            self.v_ref = self.controllers[0].low_pass_filter(self.v_ref + v_rep, self.v_ref, alpha=0.1)      
            
            u = self.controllers[i].vel_control_algorithm(
                state,
                self.v_ref[i, :2],      # full 3D velocity reference
                self.altitude_ref[i]
            )
            # Leader 1 drone follows the leader 0 to form the scale
            if i == 2 and self.tracking_flag and self.not_landing_flag:
                pos_0 = np.array(data.sensor(f'pos_{0}').data)
                ref_dir = (self.formation_controller.X_ref[i] - self.formation_controller.X_ref[0])
                pos_ref_2 = pos_0 + (self.scale * ref_dir)
                # self.leader_controller[1].Kp_pos = .01
                u = self.leader_controller[1].pos_control_algorithm(state, pos_ref_2)
                
            if i == 0 and self.tracking_flag and self.not_landing_flag:
                pos_ref = self.path_tracking.look_ahead_point(pos)
                pos_ref = self.leader_controller[0].low_pass_filter(pos_ref, pos, alpha=0.99)      
                # pos_ref = np.array([3, -2.5, 10])
                # self.leader_controller[0].Kp_pos = .1
                if not self.not_landing_flag:
                    pos_ref[2] = self.altitude_ref[0]
                u = self.leader_controller[0].pos_control_algorithm(state, pos_ref)
            
            # Apply control to actuators
            for j in range(4):
                data.actuator(f"thrust{j+1}_{i}").ctrl = u[j]

        # --- Update previous time ---
        self.t_prev = sim.data.time

