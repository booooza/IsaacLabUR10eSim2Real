"""
Unit tests for UR10e forward kinematics.

Run with:
    python -m pytest test_ur10e_kinematics.py -v
    
Or with unittest:
    python -m unittest test_ur10e_kinematics.py
"""

import unittest
import numpy as np
from scripts.kinematics.ur10e_kinematics import UR10eKinematics


class TestUR10eKinematics(unittest.TestCase):
    """Test cases for UR10e forward kinematics."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures that are used by all tests."""
        cls.fk = UR10eKinematics()
        cls.position_tolerance_m = 1e-3  # 1 mm tolerance
        cls.rotation_tolerance_rad = np.deg2rad(1.0)  # 1 degree tolerance
    
    def test_all_zeros_configuration(self):
        """Test FK with all joints at zero."""
        joints = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        pos, rot = self.fk.get_ee_pose(joints)
        roll, pitch, yaw = self.fk.rotation_matrix_to_euler_xyz(rot)

        print(f"Position at all zeros: {pos*1000} mm")
        print(f"Rotation at all zeros:\n{rot}")
        print(f"Euler angles at all zeros: roll={np.rad2deg(roll):.2f}°, pitch={np.rad2deg(pitch):.2f}°, yaw={np.rad2deg(yaw):.2f}°")
        
        # At all zeros: [ -1183.935000,  -290.700000,    60.850000,     1.570796,     0.000000,    -0.000000 ]
        expected_pos = np.array([-1183.935000, -290.700000, 60.850000]) / 1000.0  # Convert to meters
        expected_rot = np.array([np.deg2rad(90.0), 0.0, 0.0])  # roll=90°, pitch=0°, yaw=0°
        
        self.assertTrue(
            np.allclose(pos, expected_pos, atol=self.position_tolerance_m), 
            f"Position mismatch: expected {expected_pos}, got {pos}"
        )
        self.assertTrue(
            np.allclose([roll, pitch, yaw], expected_rot, atol=self.rotation_tolerance_rad),
            f"Rotation mismatch: expected {expected_rot}, got {[roll, pitch, yaw]}"
        )
    
    def test_home_configurations(self):
        """Test that different joint configurations reach the same end-effector pose.
        
        These are different IK solutions (joint configurations) that should all
        result in the same end-effector position and orientation.
        Reference: https://robodk.com/robot/Universal-Robots/UR10e
        """
        # Expected end-effector pose (in mm from RoboDK)
        expected_pos_m = np.array([687.885000, -174.150000, 913.150000]) / 1000.0  # Convert to m
        
        # Expected orientation in XYZ Euler angles (degrees from RoboDK)
        expected_euler_deg = np.array([-90.0, 0.0, -90.0])  # [roll, pitch, yaw]
        expected_euler_rad = np.deg2rad(expected_euler_deg)

        joint_configs_deg = [
            [0.000000, -90.000000, -90.000000, 0.000000, 90.000000, 0.000000],
            [0.000000, -90.000000, -90.000000, 0.000000, 90.000000, -0.000000],
            [146.096256, -27.368336, -59.892507, 87.260844, 56.096256, 180.000000],
            [146.096256, -90.000000, 90.000000, 180.000000, -56.096256, -0.000000],
            [146.096256, -84.960216, 59.892507, 25.067709, 56.096256, 180.000000],
            [0.000000, -95.039784, -59.892507, 154.932291, -90.000000, 180.000000],
            [0.000000, -90.000000, -90.000000, 0.000000, 90.000000, -0.000000],
            [0.000000, -152.631664, 59.892507, 92.739156, -90.000000, 180.000000],
            [0.000000, -176.007629, 90.000000, -93.992371, 90.000000, -0.000000],
        ]
        
        joint_configs_rad = [np.deg2rad(config) for config in joint_configs_deg]
        
        print(f"\nTesting {len(joint_configs_rad)} IK solutions:")
        print(f"Expected position (m):      {expected_pos_m}")
        print(f"Expected Euler XYZ (deg):   {expected_euler_deg}")
        print("-" * 80)
        
        for i, joints in enumerate(joint_configs_rad):
            with self.subTest(config=i):
                # Compute forward kinematics
                pos, rot = self.fk.get_ee_pose(joints)
                roll, pitch, yaw = self.fk.rotation_matrix_to_euler_xyz(rot)
                euler_computed = np.array([roll, pitch, yaw])

                print(f"\nConfig {i+1}: joints (deg) = {np.rad2deg(joints)}")
                print(f"  Position (mm):     {pos*1000}")
                print(f"  Euler XYZ (deg):   roll={np.rad2deg(roll):.2f}°, pitch={np.rad2deg(pitch):.2f}°, yaw={np.rad2deg(yaw):.2f}°")
                
                # Calculate position error
                pos_error = pos - expected_pos_m
                pos_error_norm = np.linalg.norm(pos_error)
                print(f"  Position error:    {pos_error_norm*1000:.3f} mm")

                # Calculate orientation error
                # Need to handle angle wrapping (e.g., -90° vs 270°)
                euler_error = euler_computed - expected_euler_rad
                # Normalize to [-pi, pi]
                euler_error = np.arctan2(np.sin(euler_error), np.cos(euler_error))
                euler_error_deg = np.rad2deg(euler_error)
                
                print(f"  Roll error:        {euler_error_deg[0]:.3f}°")
                print(f"  Pitch error:       {euler_error_deg[1]:.3f}°")
                print(f"  Yaw error:         {euler_error_deg[2]:.3f}°")

                # Assert position matches within tolerance
                self.assertLess(
                    pos_error_norm,
                    self.position_tolerance_m,
                    msg=f"Config {i+1}: Position error {pos_error_norm*1000:.3f}mm exceeds {self.position_tolerance_m*1000}mm tolerance.\n"
                        f"Expected: {expected_pos_m}\nGot: {pos}\nError: {pos_error}"
                )

                # Assert each Euler angle matches within tolerance
                self.assertLess(
                    abs(euler_error_deg[0]),
                    np.rad2deg(self.rotation_tolerance_rad),
                    msg=f"Config {i+1}: Roll error {euler_error_deg[0]:.3f}° exceeds {np.rad2deg(self.rotation_tolerance_rad):.1f}° tolerance.\n"
                        f"Expected: {expected_euler_deg[0]:.2f}°, Got: {np.rad2deg(roll):.2f}°"
                )
                
                self.assertLess(
                    abs(euler_error_deg[1]),
                    np.rad2deg(self.rotation_tolerance_rad),
                    msg=f"Config {i+1}: Pitch error {euler_error_deg[1]:.3f}° exceeds {np.rad2deg(self.rotation_tolerance_rad):.1f}° tolerance.\n"
                        f"Expected: {expected_euler_deg[1]:.2f}°, Got: {np.rad2deg(pitch):.2f}°"
                )
                
                self.assertLess(
                    abs(euler_error_deg[2]),
                    np.rad2deg(self.rotation_tolerance_rad),
                    msg=f"Config {i+1}: Yaw error {euler_error_deg[2]:.3f}° exceeds {np.rad2deg(self.rotation_tolerance_rad):.1f}° tolerance.\n"
                        f"Expected: {expected_euler_deg[2]:.2f}°, Got: {np.rad2deg(yaw):.2f}°"
                )
