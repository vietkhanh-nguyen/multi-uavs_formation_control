import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from plots.track_sim_data import TrackData

class PlotData:

    def plot_3d_map(self, box_obs_list, map_size, elev=25, azim=45, waypoints=None):
        """
        Plot the 3D map with rectangular obstacles.
        Optionally overlay a path/waypoints.

        Parameters
        ----------
        elev : float
            Elevation angle for 3D view.
        azim : float
            Azimuth angle for 3D view.
        waypoints : np.ndarray or list of (x, y, z)
            Optional path points to overlay.
        """
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # -----------------------------
        # Plot rectangular obstacles
        # -----------------------------
        for obs in box_obs_list:   # each = [xmin, xmax, ymin, ymax, zmin, zmax]
            xmin, xmax, ymin, ymax, zmin, zmax = obs

            # 8 vertices of the box
            vertices = np.array([
                [xmin, ymin, zmin],
                [xmax, ymin, zmin],
                [xmax, ymax, zmin],
                [xmin, ymax, zmin],
                [xmin, ymin, zmax],
                [xmax, ymin, zmax],
                [xmax, ymax, zmax],
                [xmin, ymax, zmax],
            ])

            faces = [
                [vertices[j] for j in [0,1,2,3]],  # bottom
                [vertices[j] for j in [4,5,6,7]],  # top
                [vertices[j] for j in [0,1,5,4]],  # front
                [vertices[j] for j in [2,3,7,6]],  # back
                [vertices[j] for j in [1,2,6,5]],  # right
                [vertices[j] for j in [0,3,7,4]],  # left
            ]

            box = Poly3DCollection(faces, alpha=0.4, edgecolor='k')
            box.set_facecolor((0.3, 0.3, 0.8, 0.4))  # bluish
            ax.add_collection3d(box)

        # -----------------------------
        # Plot waypoints if provided
        # -----------------------------
        if waypoints is not None:
            waypoints = np.array(waypoints)
            ax.plot(waypoints[:,0], waypoints[:,1], waypoints[:,2], color='r', linewidth=2, label='Path')
            ax.scatter(waypoints[0,0], waypoints[0,1], waypoints[0,2], color='g', s=60, label='Start')
            ax.scatter(waypoints[-1,0], waypoints[-1,1], waypoints[-1,2], color='b', s=60, label='Goal')

        # -----------------------------
        # Map bounding box
        # -----------------------------
        xmin, xmax, ymin, ymax, zmin, zmax = map_size
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_zlim(zmin, zmax)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("3D Map Environment")
        ax.view_init(elev=elev, azim=azim)
        if waypoints is not None:
            ax.legend()

        plt.tight_layout()
        plt.show()


    def plot_sim_1_drone(self, tracked_data, final_time):

        fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        tracked_data = np.array(tracked_data)

        t = np.linspace(0, final_time, len(tracked_data))
        # Subplot 1: Pitch angle tracking
        axs[0].plot(t, tracked_data[:, 3], label='Roll', color='tab:orange', linewidth=2)
        axs[0].plot(t, tracked_data[:, 4], label='Pitch', color='tab:olive', linestyle='--', linewidth=2)
        axs[0].plot(t, tracked_data[:, 5], label='Yaw', color='tab:purple', linestyle='--', linewidth=2)
        axs[0].set_ylabel("Angle")
        axs[0].set_title("Angle Tracking Over Time")
        axs[0].legend()
        axs[0].grid(True)

        # Subplot 2: Position
        axs[1].plot(t, tracked_data[:, 0], label='X Position', color='tab:red', linewidth=2)
        axs[1].plot(t, tracked_data[:, 1], label='Y Position', color='tab:green', linewidth=2)
        axs[1].plot(t, tracked_data[:, 2], label='Z Position', color='tab:blue', linewidth=2)
        axs[1].set_xlabel("Timestep")
        axs[1].set_ylabel("Position")
        axs[1].set_title("Position Tracking Over Time")
        axs[1].legend()
        axs[1].grid(True)

        plt.tight_layout()
        plt.show()

    def plot_sim_multi_drones(self, tracker:TrackData):
        """
        Plot multi-UAV simulation results:
        - 3D centroid trajectory and waypoints
        - Initial and final positions
        - Scale over time
        - Bearing errors over time
        """

        t = tracker.t
        centroid = tracker.centroid
        scale = tracker.scale
        scale_ref = tracker.scale_ref
        n_edges = tracker.e_ij_norms.shape[0]
        e_norms = tracker.e_ij_norms
        n_drones = tracker.n_drones

        # -----------------------------------
        # 3D plot: centroid, initial & final positions, waypoints
        # -----------------------------------
        fig = plt.figure(figsize=(12, 5))
        ax1 = fig.add_subplot(121, projection='3d')
        
        # Plot centroid trajectory
        ax1.plot(centroid[0], centroid[1], centroid[2], color='b', linewidth=2, label='Centroid')

        # Plot initial positions
        print(tracker.state_full.shape)

        
        print(tracker.moment_operation)
        for i in range(len(tracker.moment_operation)):
            pos = tracker.state_full[:, :3, tracker.moment_operation[i]]
            ax1.scatter(pos[:,0], pos[:,1], pos[:,2], s=60)

        # Optional: waypoints
        if hasattr(tracker, "waypoints") and tracker.waypoints is not None:
            wp = np.array(tracker.waypoints)
            ax1.plot(wp[:,0], wp[:,1], wp[:,2], color='k', linewidth=2, linestyle='--', label='Waypoints')

        x_limits = [np.min(tracker.state_full[:, 0, :]), np.max(tracker.state_full[:, 0, :])]
        y_limits = [np.min(tracker.state_full[:, 1, :]), np.max(tracker.state_full[:, 1, :])]
        z_limits = [np.min(tracker.state_full[:, 2, :]), np.max(tracker.state_full[:, 2, :])]

        # Find the middle and max range
        x_middle = np.mean(x_limits)
        y_middle = np.mean(y_limits)
        z_middle = np.mean(z_limits)

        max_range = np.max([x_limits[1]-x_limits[0], y_limits[1]-y_limits[0], z_limits[1]-z_limits[0]]) / 2

        # Set equal limits around the middle
        ax1.set_xlim(x_middle - max_range, x_middle + max_range)
        ax1.set_ylim(y_middle - max_range, y_middle + max_range)
        ax1.set_zlim(z_limits[0], z_limits[0] + 2*max_range)

        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        ax1.set_zlabel("Z")
        ax1.set_title("UAV Centroid & Positions")
        ax1.legend()
        ax1.grid(True)


        # -----------------------------------
        # Subplots: scale and bearing errors
        # -----------------------------------
        ax2 = fig.add_subplot(222)
        ax2.plot(t, tracker.centroid[0, :], color='r', linewidth=2,label="X")
        ax2.plot(t, tracker.centroid[1, :], color='g', linewidth=2,label="Y")
        ax2.plot(t, tracker.centroid[2, :], color='b', linewidth=2,label="Z")
        ax2.legend()
        
        ax2.set_xlabel("Time [s]")
        ax2.set_ylabel("Position")
        ax2.set_title("Centroid Position Over Time")
        ax2.grid(True)

        ax3 = fig.add_subplot(224)
        for k in range(n_edges):
            ax3.plot(t, e_norms[k], linewidth=1)
        ax3.set_xlabel("Time [s]")
        ax3.set_ylabel("Bearing error ||e_ij||")
        ax3.set_title("Bearing Errors Over Time")
        ax3.grid(True)

        plt.tight_layout()
        plt.show()
