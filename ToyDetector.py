import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

class ToyDetector:
    def __init__(self):
        # Original cylindrical detector specifications
        self.radius = 0.2  # 20 cm
        self.eta_min = -1.5
        self.eta_max = 1.5
        self.phi_segments = 8
        self.eta_segments = 12
        
        # Planar detectors specifications
        self.z_max = self.radius * np.tanh(self.eta_max)  # z_max based on the cylindrical detector's eta_max
        self.z_min = self.radius * np.tanh(self.eta_min)  # z_min based on the cylindrical detector's eta_min
        self.planar_z1 = 1.2 * self.z_max  # Planar detector at z = 1.2 * z_max
        self.planar_z2 = 1.2 * self.z_min  # Planar detector at z = 1.2 * z_min
        self.planar_side_length = 0.4  # 40 cm side length
        self.planar_segments = 16  # 16x16 tiles

        # Create modules for the detectors
        self.modules = self.create_modules()

    def create_modules(self):
        """Creates detector modules for both the cylindrical and planar detectors."""
        modules = []
        module_id = 0  # Initialize module ID
        
        # Cylindrical detector modules
        eta_step = (self.eta_max - self.eta_min) / self.eta_segments
        phi_step = 2 * np.pi / self.phi_segments
        
        for i in range(self.eta_segments):
            for j in range(self.phi_segments):
                module = {
                    'module_ID': module_id,  # Assign a unique module ID
                    'eta_min': self.eta_min + i * eta_step,
                    'eta_max': self.eta_min + (i + 1) * eta_step,
                    'phi_min': j * phi_step,
                    'phi_max': (j + 1) * phi_step,
                    'hit': False,
                    'type': 'cylinder'
                }
                modules.append(module)
                module_id += 1  # Increment module ID

        # Planar detector modules (two detectors)
        planar_step = self.planar_side_length / self.planar_segments
        for k, z_pos in enumerate([self.planar_z1, self.planar_z2]):
            for i in range(self.planar_segments):
                for j in range(self.planar_segments):
                    module = {
                        'module_ID': module_id,  # Assign a unique module ID
                        'x_min': -0.5 * self.planar_side_length + i * planar_step,
                        'x_max': -0.5 * self.planar_side_length + (i + 1) * planar_step,
                        'y_min': -0.5 * self.planar_side_length + j * planar_step,
                        'y_max': -0.5 * self.planar_side_length + (j + 1) * planar_step,
                        'z_pos': z_pos,
                        'hit': False,
                        'type': 'planar',
                        'detector_id': k+1
                    }
                    modules.append(module)
                    module_id += 1  # Increment module ID
        
        return modules

    def reset_hits(self):
        """Resets the hit status of all detector modules."""
        for module in self.modules:
            module['hit'] = False

    def set_hit_status(self, specific_module_ID):
        """
        Sets the hit status to True for a specific module identified by its module_ID.
        
        Parameters:
        specific_module_ID (int): The ID of the module to search for in the module array.
        """
        for module in self.modules:  # Loop over the module array
            if module['module_ID'] == specific_module_ID:
                module['hit'] = True  # Set hit status to True
                break  # Exit the loop once the module is found and updated

    def identify_module(self, eta=None, phi=None, x=None, y=None, z=None):
        """Identify which module a particle belongs to based on its eta, phi, x, y, and z coordinates."""
        for module in self.modules:
            if module['type'] == 'cylinder' and eta is not None and phi is not None:
                if module['eta_min'] <= eta < module['eta_max'] and module['phi_min'] <= phi < module['phi_max']:
                    return module['module_ID']
            elif module['type'] == 'planar' and x is not None and y is not None and z is not None:
                if module['x_min'] <= x < module['x_max'] and module['y_min'] <= y < module['y_max'] and np.isclose(z, module['z_pos']):
                    return module['module_ID']
        return None  # Particle is outside the detector coverage

    def compute_crossed_detector_elements(self, df, event_number):
        """Computes the crossed detector elements for a given event number."""
        self.reset_hits()  # Reset the hit status for all modules before processing a new event

        crossed_elements = []

        # Filter the dataframe for the specified event number
        event_tracks = df[df['Event ID'] == event_number]

        for _, track in event_tracks.iterrows():
            px = track['px']
            py = track['py']
            pz = track['pz']

            # Compute the rapidity eta and azimuthal angle phi
            p_total = np.sqrt(px**2 + py**2 + pz**2)
            eta = 0.5 * np.log((p_total + pz) / (p_total - pz))
            phi = np.arctan2(py, px)

            # Determine if the particle hits the cylindrical detector
            module_id = self.identify_module(eta=eta, phi=phi)
            if module_id is not None:
                self.set_hit_status(module_id)
                crossed_elements.append(module_id)

            # Check the direction of pz and only compute crossings for the appropriate planar detector
            if pz > 0:
                z_pos = self.planar_z1  # Right side planar detector
                t = z_pos / pz  # Compute the parameter t where the particle intersects the z-plane
                x_pos = t * px
                y_pos = t * py
                module_id = self.identify_module(x=x_pos, y=y_pos, z=z_pos)
                if module_id is not None:
                    self.set_hit_status(module_id)
                    crossed_elements.append(module_id)

            elif pz < 0:
                z_pos = self.planar_z2  # Left side planar detector
                t = z_pos / pz  # Compute the parameter t where the particle intersects the z-plane
                x_pos = t * px
                y_pos = t * py
                module_id = self.identify_module(x=x_pos, y=y_pos, z=z_pos)
                if module_id is not None:
                    self.set_hit_status(module_id)
                    crossed_elements.append(module_id)

        return crossed_elements

    def visualize_detector_3d(self):
        """Visualizes the full toy detector geometry in 3D."""
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        for module in self.modules:
            if module['type'] == 'cylinder':
                z_min = self.radius * np.tanh(module['eta_min'])
                z_max = self.radius * np.tanh(module['eta_max'])
                theta = [module['phi_min'], module['phi_min'], module['phi_max'], module['phi_max'], module['phi_min']]
                r = [self.radius] * 5  # Fixed radius for the cylindrical surface
                
                # Convert polar coordinates to Cartesian for 3D plotting
                x = np.multiply(r, np.cos(theta))
                y = np.multiply(r, np.sin(theta))
                z = [z_min, z_max, z_max, z_min, z_min]

                # Create vertices for the 3D plot
                vertices = [list(zip(x, y, z))]
                poly = Poly3DCollection(vertices, alpha=0.3, facecolor='blue')
                ax.add_collection3d(poly)

            elif module['type'] == 'planar':
                x = [module['x_min'], module['x_max'], module['x_max'], module['x_min'], module['x_min']]
                y = [module['y_min'], module['y_min'], module['y_max'], module['y_max'], module['y_min']]
                z = [module['z_pos']] * 5

                # Create vertices for the 3D plot
                vertices = [list(zip(x, y, z))]
                poly = Poly3DCollection(vertices, alpha=0.3, facecolor='blue')
                ax.add_collection3d(poly)

        # Set labels and limits
        ax.set_xlabel('X (cos(phi))')
        ax.set_ylabel('Y (sin(phi))')
        ax.set_zlabel('Z (beam axis)')
        ax.set_title('3D Visualization of Toy Detector Geometry')

        # Adjust limits to ensure full view of the geometry
        ax.set_xlim([-0.25, 0.25])
        ax.set_ylim([-0.25, 0.25])
        ax.set_zlim([-0.25, 0.25])

        plt.show()

    def visualize_event(self, df, event_number, dashed=False, line_color='green'):
        """Visualizes the toy detector geometry in 3D with hit modules and particle tracks."""
        crossed_elements = self.compute_crossed_detector_elements(df, event_number)

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        for module in self.modules:
            color = 'red' if module['module_ID'] in crossed_elements else 'blue'

            if module['type'] == 'cylinder':
                z_min = self.radius * np.tanh(module['eta_min'])
                z_max = self.radius * np.tanh(module['eta_max'])
                theta = [module['phi_min'], module['phi_min'], module['phi_max'], module['phi_max'], module['phi_min']]
                r = [self.radius] * 5  # Fixed radius for the cylindrical surface
                
                # Convert polar coordinates to Cartesian for 3D plotting
                x = np.multiply(r, np.cos(theta))
                y = np.multiply(r, np.sin(theta))
                z = [z_min, z_max, z_max, z_min, z_min]

                # Create vertices for the 3D plot
                vertices = [list(zip(x, y, z))]
                poly = Poly3DCollection(vertices, alpha=0.3, facecolor=color)
                ax.add_collection3d(poly)

            elif module['type'] == 'planar':
                x = [module['x_min'], module['x_max'], module['x_max'], module['x_min'], module['x_min']]
                y = [module['y_min'], module['y_min'], module['y_max'], module['y_max'], module['y_min']]
                z = [module['z_pos']] * 5

                # Create vertices for the 3D plot
                vertices = [list(zip(x, y, z))]
                poly = Poly3DCollection(vertices, alpha=0.3, facecolor=color)
                ax.add_collection3d(poly)

        # Plot the particle tracks
        for _, track in df[df['Event ID'] == event_number].iterrows():
            px = track['px']
            py = track['py']
            pz = track['pz']
            ctau = track['c*tau']
            
            # Compute the endpoint of the track based on its lifetime
            p_total = np.sqrt(px**2 + py**2 + pz**2)
            endpoint = {
                'x': px / p_total * ctau,
                'y': py / p_total * ctau,
                'z': pz / p_total * ctau
            }
            
            # Plot the track as a line
            line_style = '--' if dashed else '-'
            ax.plot([0, endpoint['x']], [0, endpoint['y']], [0, endpoint['z']], line_style, color=line_color)

        # Set labels and limits
        ax.set_xlabel('X (cos(phi))')
        ax.set_ylabel('Y (sin(phi))')
        ax.set_zlabel('Z (beam axis)')
        ax.set_title(f'3D Visualization of Event {event_number}')
        
        # Adjust limits to ensure full view of the geometry
        ax.set_xlim([-0.25, 0.25])
        ax.set_ylim([-0.25, 0.25])
        ax.set_zlim([-0.25, 0.25])

        plt.show()

# Example usage (uncomment these lines to run):
# detector = ToyDetector()
# detector.visualize_detector_3d()
