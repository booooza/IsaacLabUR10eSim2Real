import numpy as np

class UR10eKinematics:
    """
    Forward kinematics for UR10e using official DH parameters.
    
    Source: Universal Robots UR10e/UR12e Technical Specifications
    https://www.universal-robots.com/articles/ur/application-installation/dh-parameters-for-calculations-of-kinematics-and-dynamics/

    Convention: Standard DH (Denavit-Hartenberg) parameters
    """
    
    def __init__(self):
        # Official UR10e DH parameters from Universal Robots
        # [a (m), d (m), alpha (rad)]
        # Note: theta is the joint variable
        
        self.dh_params = np.array([
            [0.0,       0.1807,     np.pi/2],   # Joint 1
            [-0.6127,   0.0,        0.0],       # Joint 2  
            [-0.57155,  0.0,        0.0],       # Joint 3
            [0.0,       0.17415,    np.pi/2],   # Joint 4
            [0.0,       0.11985,    -np.pi/2],  # Joint 5
            [0.0,       0.11655,    0.0],       # Joint 6
        ])
        
        # Store for reference
        self.a = self.dh_params[:, 0]
        self.d = self.dh_params[:, 1]
        self.alpha = self.dh_params[:, 2]

    def dh_transform(self, a, d, alpha, theta):
        """
        Compute transformation matrix from standard DH parameters.
        
        Standard DH Convention (Craig):
        T_i = Rot_z(theta_i) * Trans_z(d_i) * Trans_x(a_i) * Rot_x(alpha_i)
        
        Args:
            a: link length
            d: link offset
            alpha: link twist
            theta: joint angle
            
        Returns:
            4x4 homogeneous transformation matrix
        """
        ct = np.cos(theta)
        st = np.sin(theta)
        ca = np.cos(alpha)
        sa = np.sin(alpha)
        
        T = np.array([
            [ct,    -st*ca,   st*sa,   a*ct],
            [st,     ct*ca,  -ct*sa,   a*st],
            [0,      sa,      ca,      d   ],
            [0,      0,       0,       1   ]
        ])
        return T
    
    def forward_kinematics(self, joint_angles):
        """
        Compute forward kinematics given joint angles.
        
        Args:
            joint_angles: Array of 6 joint angles in radians [θ1, θ2, θ3, θ4, θ5, θ6]
            
        Returns:
            4x4 transformation matrix from base to end-effector (wrist_3_link frame)
        """
        if len(joint_angles) != 6:
            raise ValueError("joint_angles must have 6 elements")
        
        T = np.eye(4)
        
        for i in range(6):
            a = self.a[i]
            d = self.d[i]
            alpha = self.alpha[i]
            theta = joint_angles[i]
            
            T_i = self.dh_transform(a, d, alpha, theta)
            T = T @ T_i
        
        return T
    
    def get_ee_pose(self, joint_angles):
        """
        Get end-effector position and orientation.
        
        Args:
            joint_angles: Array of 6 joint angles in radians
            
        Returns:
            position: 3D position vector [x, y, z] in meters
            rotation_matrix: 3x3 rotation matrix
        """
        T = self.forward_kinematics(joint_angles)
        
        position = T[:3, 3]
        rotation_matrix = T[:3, :3]
        
        return position, rotation_matrix
    
    def rotation_matrix_to_euler_xyz(self, R):
        """
        Convert rotation matrix to Euler angles (XYZ convention).
        
        Args:
            R: 3x3 rotation matrix
            
        Returns:
            [roll, pitch, yaw] in radians
        """
        sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
        
        singular = sy < 1e-6
        
        if not singular:
            roll = np.arctan2(R[2, 1], R[2, 2])
            pitch = np.arctan2(-R[2, 0], sy)
            yaw = np.arctan2(R[1, 0], R[0, 0])
        else:
            roll = np.arctan2(-R[1, 2], R[1, 1])
            pitch = np.arctan2(-R[2, 0], sy)
            yaw = 0
        
        return np.array([roll, pitch, yaw])
