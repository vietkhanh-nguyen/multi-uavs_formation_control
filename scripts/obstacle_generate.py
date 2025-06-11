import xml.etree.ElementTree as ET

# Define obstacle data
all_obs_center = [[5, 4, 3], [8, 3, 1]]
all_obs_vel = [[0, -0.2, 0.2], [0, 0, 0]]
obs_radius = 2
obs_height = 1.0  # Set the height of the obstacle cylinders

def generate_obstacles_xml(obs_centers, obs_vels, radius, height=1.0):
    mujoco = ET.Element('mujoco', attrib={'model': 'obstacles'})
    worldbody = ET.SubElement(mujoco, 'worldbody')
    actuator = ET.SubElement(mujoco, 'actuator')

    for i, (center, vel) in enumerate(zip(obs_centers, obs_vels)):
        x, y, z = center
        vx, vy, vz = vel

        body = ET.SubElement(worldbody, 'body', name=f'obstacle_{i}', pos=f"{x} {y} {z}")

        # Add slide joints for motion in x, y, z directions
        # ET.SubElement(body, 'joint', name=f'obstacle_{i}_x', type="slide", axis="1 0 0")
        # ET.SubElement(body, 'joint', name=f'obstacle_{i}_y', type="slide", axis="0 1 0")
        # ET.SubElement(body, 'joint', name=f'obstacle_{i}_z', type="slide", axis="0 0 1")

        # Add cylinder geom
        ET.SubElement(
            body, 'geom',
            type='sphere',
            size=f"{radius}",
            mass="1.0",
            rgba="1 0 0 0.5"
        )

        # # Add velocity actuators
        # ET.SubElement(actuator, 'velocity', joint=f'obstacle_{i}_x', kv="1000", ctrlrange="-10 10")
        # ET.SubElement(actuator, 'velocity', joint=f'obstacle_{i}_y', kv="1000", ctrlrange="-10 10")
        # ET.SubElement(actuator, 'velocity', joint=f'obstacle_{i}_z', kv="1000", ctrlrange="-10 10")

    # Write to XML string and file
    tree = ET.ElementTree(mujoco)
    tree.write("obstacles.xml", encoding="utf-8", xml_declaration=True)

# Run generator
generate_obstacles_xml(all_obs_center, all_obs_vel, obs_radius, obs_height)
