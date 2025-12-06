# Execution Module

`execution/` contains the main entry point for running all drone simulation scenarios in this project.  
Each scenario uses MuJoCo, custom multi-agent controllers, and various path-planning and consensus algorithms to simulate drone formation, tracking, and navigation behaviors.

---

## Overview

The main file here is `main.py`


This script allows you to run different simulation configurations by calling one of the scenario functions:

- Bearing-based Center Tracking Consensus
- Bearing-based Tracking Consensus
- Bearing-based Formation Consensus
- Single-Drone Tracking

Each scenario loads a MuJoCo scene, builds drone models dynamically, initializes controllers, and launches a simulation (with optional rendering or video export).

---

## Available Scenarios

### 1. Function: `mjc_sim_scenario_drone_tracking()`
- 1 drone tracking waypoints only
- Ideal for verify the waypoint tracking algorithm
- Basic PID-based quadcopter tracking  


### 2. Function: `mjc_sim_scenario_bearing_based()`
- Formation control only
- Using to verify the consensus control
- Scenario: take-off → icosahedron → rectangle formation → landing

### 3. Function: `mjc_sim_scenario_bearing_center_tracking_based()`
- My complete framework (integrating every thing)
- Scenario: intial formation → icosahedron → resize → follow waypoint → rectangle formation
- Automatic takeoff and landing sequence
- 2 virtual leader agents

### 4. Function: `mjc_sim_scenario_bearing_tracking_based()`s 
- Mostly using to compare with the complete framework
- Scenario: intial formation → icosahedron → pursuit reference waypoint → rectangle formation
- Automatic takeoff and landing sequence
- 2 UAVs as leader agents




