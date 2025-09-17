import rvo2

import numpy as np

class PIDController:

    def __init__(self, Kp, Ki, Kd, time_step):
        
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.int = 0
        self.dt = time_step
        self.prev_e = None
        self.prev_de = None

    def low_pass_filter(self, raw_de, pre_de):
        alpha = 0.2
        return (1 - alpha)*pre_de + alpha*raw_de

    def algorithm(self, e, raw_de=None):
        if self.prev_e is None:
            self.prev_e = e
            self.prev_de = np.zeros_like(e)
        if raw_de is None:
            raw_de = (e - self.prev_e) / self.dt
        de = self.low_pass_filter(raw_de, self.prev_de)
        self.int = np.clip(self.int + e * self.dt, -200, 200)

        self.prev_e = e
        self.prev_de = de
        return self.Kp * e + self.Ki * self.int + self.Kd * de

class LeaderFollowerController:

    def __init__(self, time_step, offset):
        """
        timeStep        The time step of the simulation.
                        Must be positive.
        neighborDist    The default maximum distance (center point
                        to center point) to other agents a new agent
                        takes into account in the navigation. The
                        larger this number, the longer the running
                        time of the simulation. If the number is too
                        low, the simulation will not be safe. Must be
                        non-negative.
        maxNeighbors    The default maximum number of other agents a
                        new agent takes into account in the
                        navigation. The larger this number, the
                        longer the running time of the simulation.
                        If the number is too low, the simulation
                        will not be safe.
        timeHorizon     The default minimal amount of time for which
                        a new agent's velocities that are computed
                        by the simulation are safe with respect to
                        other agents. The larger this number, the
                        sooner an agent will respond to the presence
                        of other agents, but the less freedom the
                        agent has in choosing its velocities.
                        Must be positive.
        timeHorizonObst The default minimal amount of time for which
                        a new agent's velocities that are computed
                        by the simulation are safe with respect to
                        obstacles. The larger this number, the
                        sooner an agent will respond to the presence
                        of obstacles, but the less freedom the agent
                        has in choosing its velocities.
                        Must be positive.
        radius          The default radius of a new agent.
                        Must be non-negative.
        maxSpeed        The default maximum speed of a new agent.
                        Must be non-negative.
        velocity        The default initial two-dimensional linear
                        velocity of a new agent (optional).

        ORCA first uses neighborDist and maxNeighbors to find neighbors that need to be taken into account.
        Here set them to be large enough so that all agents will be considered as neighbors.
        Time_horizon should be set that at least it's safe for one time step

        In this work, obstacles are not considered. So the value of time_horizon_obst doesn't matter.

        """
        self.num_agent = len(offset) + 1
        self.offset = offset
        self.safety_space = 0.2
        self.neighbor_dist = 5
        self.max_neighbors = 10
        self.time_horizon = 2
        self.time_horizon_obst = 2
        self.radius = 0.4
        self.time_step = time_step
        self.max_speed = 2
        self.controllers = []
        self.sim = None

    def rotate(self, follower_offset, leader_yaw):
        R = np.array([
            [np.cos(leader_yaw), -np.sin(leader_yaw)],
            [np.sin(leader_yaw), np.cos(leader_yaw)]
        ])
        return R@follower_offset
    
    def cal_all_agent_offset(self, leader_q):
        leader_pos = leader_q[0:2]
        leader_yaw = leader_q[5]
        desired_lineup = [leader_pos]
        for i in range(self.num_agent - 1):
            desired_pos = leader_pos + self.rotate(self.offset[i], leader_yaw)
            desired_lineup.append(desired_pos)
        return desired_lineup

    def algorithm(self, all_agent_state):
        # agent_state = [agent_pos_x, agent_pos_y, agent_vel_x, agent_vel_y]
        # agent_offset = [agent_offset_x, agent_offset_y]
        all_agent_state = [tuple(arr.tolist()) for arr in all_agent_state]
        leader_state = all_agent_state[0]
        if self.sim is None:
            params = self.neighbor_dist, self.max_neighbors, self.time_horizon, self.time_horizon_obst
            self.sim = rvo2.PyRVOSimulator(self.time_step, *params, self.radius, self.max_speed)
            for i, agent_state in enumerate(all_agent_state):
                self.sim.addAgent(agent_state[0:2], *params, self.radius + self.safety_space,
                                  self.max_speed, agent_state[6:8])
            Kp = 1.0
            Ki = 0.5
            Kd = 2.0
            for i in range(len(all_agent_state)):
                controller = PIDController(Kp, Ki, Kd, self.time_step)
                self.controllers.append(controller)
        else:
            for i, agent_state in enumerate(all_agent_state):
                self.sim.setAgentPosition(i, agent_state[0:2])
                # relative_vel = tuple(np.array(agent_state[6:8]) - np.array(leader_state[6:8]))
                self.sim.setAgentVelocity(i, agent_state[6:8])
                # self.sim.setAgentVelocity(i, relative_vel)

        all_agent_offset = self.cal_all_agent_offset(leader_state)
        i = 0
        for agent_state, agent_offset in zip(all_agent_state, all_agent_offset):
            # unknown goal position of other humans
            error = np.array([agent_offset[0] - agent_state[0], agent_offset[1] - agent_state[1]])

            velocity = self.controllers[i].algorithm(error, -np.array(agent_state[6:8]))
            speed = np.linalg.norm(velocity)
            pref_vel = velocity / speed * self.max_speed if speed > self.max_speed  else velocity
            self.sim.setAgentPrefVelocity(i, tuple(pref_vel))
            i += 1

        self.sim.doStep()

        next_swarm_vel = []
        for i in range(self.num_agent):
            # unknown goal position of other humans
            agent_vel = self.sim.getAgentVelocity(i)
            next_swarm_vel.append(agent_vel)

        return next_swarm_vel



if __name__ == "__main__":
    # Example: 1 leader at (10, 5) facing 90 degrees (pi/2 radians)
    leader_state = np.array([10.0, 5.0, np.pi / 2])  # x, y, yaw

    # Two followers: 2 meters behind, 1 meter left/right in leader's frame
    offsets = [
        np.array([-2.0, -1.0]),
        np.array([-2.0,  1.0])
    ]

    controller = LeaderFollowerController(offsets)
    follower_targets = controller.solve(leader_state)

    # print(follower_targets)

    for i, pos in enumerate(follower_targets):
        print(f"Follower {i+1} target position: {pos}")