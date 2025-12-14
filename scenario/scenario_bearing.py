import numpy as np
from controls.quadcopter_controller import QuadcopterPIDController
from controls.consensus_controller import MultiAgentConsensus

class ScenarioBearingbasedConsensus:

    def __init__(self):
        self.name = "Drones Formation using Bearing-based Consensus Algorithm"
        self.e_bearing_tol = 1

    def init(self, sim, model, data):

        sim.cam.azimuth = -0.87
        sim.cam.elevation = -25
        sim.cam.distance =  16
        sim.cam.lookat =np.array([ 0.0 , 0.0 , 0.0 ])
        
        self.controllers = []
        self.formation_controller = MultiAgentConsensus(sim.num_drones, K=1)

        for _ in range(sim.num_drones):
            self.controllers.append(QuadcopterPIDController(sim.time_step))

        # PID for path following (để riêng)
        self.follow_controller = QuadcopterPIDController(sim.time_step)
        self.t_prev = sim.data.time
        self.operation_mode = "take_off"

    def update(self, sim, model, data):

        n_drones = sim.num_drones
        dt = sim.data.time - self.t_prev
        # --- Read current states ---
        pos_full = np.zeros((n_drones, 3))
        quat_full = np.zeros((n_drones, 4))
        linvel_full = np.zeros((n_drones, 3))
        angvel_full = np.zeros((n_drones, 3))
        state_full = np.zeros((n_drones, 13))
        acc_full = np.zeros((n_drones, 3))        
        
        for i in range(n_drones):
            
            pos_full[i, :] = np.array(data.sensor(f'pos_{i}').data)
            quat_full[i, :] = np.array(data.sensor(f'quat_{i}').data)
            linvel_full[i, :] = np.array(data.sensor(f'vel_{i}').data)
            angvel_full[i, :] = np.array(data.sensor(f'gyro_{i}').data)

            vel = np.hstack((linvel_full[i, :], angvel_full[i, :]))
            state_full[i, :] = np.concatenate([pos_full[i, :], quat_full[i, :], vel])

            body_acc = np.array(data.sensor(f'acc_{i}').data) 
            acc_full[i, :] = self.controllers[i].linear_acc_world(quat_full[i, :], body_acc)
              
        e_bearing = self.formation_controller.compute_bearing_error(pos_full)
        e_bearing_norm = np.linalg.norm(e_bearing)
        sim.cam.lookat = np.mean(pos_full, axis=0)
        sim.cam.azimuth += .1 

        match self.operation_mode:

            case "take_off":
                self.v_ref = np.zeros_like(pos_full)
                self.altitude_ref = 5 * np.ones(sim.num_drones)
                
                if np.all(np.abs(pos_full[:, 2] - self.altitude_ref) < self.e_bearing_tol):
                    self.operation_mode = "formation_icosahedron"
                    self.altitude_ref = pos_full[:, 2]
                    

            case "formation_icosahedron":
                control_consensus = self.formation_controller.consensus_law(pos_full)
                self.v_ref = control_consensus       
                self.altitude_ref += self.v_ref[:, 2] * dt

                if e_bearing_norm < self.e_bearing_tol:
                    self.operation_mode = "formation_rectangle"
                    self.altitude_ref = pos_full[:, 2]
                    self.formation_controller._init_states("rectangle")
                
            case "formation_rectangle":
                control_consensus = self.formation_controller.consensus_law(pos_full)
                self.v_ref = control_consensus    
                self.altitude_ref += self.v_ref[:, 2] * dt

                if (e_bearing_norm < self.e_bearing_tol):
                    self.operation_mode = "landing"

            case "landing":
                self.v_ref = np.zeros_like(pos_full)
                self.altitude_ref = np.zeros(n_drones)

        for i in range(sim.num_drones):

            pos = np.array(data.sensor(f'pos_{i}').data)
            quat = np.array(data.sensor(f'quat_{i}').data)
            linvel = data.sensor(f'vel_{i}').data
            angvel = data.sensor(f'gyro_{i}').data
            vel = np.hstack((linvel, angvel))
            state = np.concatenate([pos, quat, vel])

            u = self.controllers[i].vel_control_algorithm(
                state,
                self.v_ref[i, :2],
                self.altitude_ref[i],
            )

            for j in range(4):
                data.actuator(f"thrust{j+1}_{i}").ctrl = u[j]

        self.t_prev = sim.data.time

    def finish(self):
        return