import cv2
import numpy as np
import pybullet as p
import pybullet_data
import time
from CameraCalibration import CameraCalibration
from DetectingMarkers import DetectingMarkers
from PixelToRealWorldConversion import PixelToRealWorldConversion

class RoboticTaskDemo:
    def __init__(self):
        self.camera_matrix = None
        self.dist_coeffs = None
        self.corners = None

    def run(self):
        # Step 1: Perform Camera Calibration
        print("Running Camera Calibration...")
        calibrator = CameraCalibration()
        if not calibrator.calibrate():
            print("Camera calibration failed. Exiting.")
            return
        self.camera_matrix, self.dist_coeffs = calibrator.get_calibration_data()

        # Step 2: Detect Markers in an Image
        print("\nRunning Marker Detection...")
        detector = DetectingMarkers()
        image_path = "checkerboards/Checkerboard-A4-25mm-10x7.jpg"  # Replace with your image path
        if not detector.detect(image_path):
            print("Marker detection failed. Exiting.")
            return
        self.corners = detector.get_corners()

        # Step 3: Convert Pixel Coordinates to Real-World Coordinates
        print("\nRunning Pixel to Real-World Conversion...")
        square_size = 0.025  # Each square is 25 mm = 0.025 meters
        checkerboard_size = (10, 7)  # Inner corners of the checkerboard

        # Prepare object points in the real-world coordinate system
        objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2) * square_size

        # Find the rotation and translation vectors
        ret, rvecs, tvecs = cv2.solvePnP(objp, self.corners, self.camera_matrix, self.dist_coeffs)

        if not ret:
            print("Failed to solve PnP. Exiting.")
            return

        # Convert a pixel coordinate to real-world coordinates (example: first corner)
        pixel_x, pixel_y = self.corners[0][0]
        converter = PixelToRealWorldConversion(self.camera_matrix, self.dist_coeffs)
        real_world_coords = converter.convert((pixel_x, pixel_y), rvecs, tvecs)

        print(f"Pixel coordinates: ({pixel_x}, {pixel_y})")
        print(f"Real-world coordinates: {real_world_coords} meters")

        # Step 4: Simulate Robotic Task
        print("\nSimulating Robotic Task...")

        # Initialize PyBullet
        physicsClient = p.connect(p.GUI)  # Connect to the PyBullet GUI
        p.setAdditionalSearchPath(pybullet_data.getDataPath())  # Set the data path for PyBullet assets
        p.setGravity(0, 0, -9.81)  # Set gravity

        # Load the plane and robot
        planeId = p.loadURDF("plane.urdf")  # Load a plane
        robotId = p.loadURDF("r2d2.urdf", [0, 0, 1])  # Load a robot (e.g., R2D2)

        # Load an object to pick up
        objectId = p.loadURDF("cube_small.urdf", [1, 1, 1])  # Load a small cube as the object

        # Move the robot to the real-world coordinates
        p.resetBasePositionAndOrientation(robotId, real_world_coords, [0, 0, 0, 1])

        # Simulate picking up the object (attach the object to the robot)
        p.createConstraint(robotId, -1, objectId, -1, p.JOINT_FIXED, [0, 0, 0], [0, 0, 0], real_world_coords)

        # Run the simulation for a few seconds
        for _ in range(1000):
            p.stepSimulation()
            time.sleep(1.0 / 240.0)  # Simulate at 240 Hz

        # Disconnect from PyBullet
        p.disconnect()

if __name__ == "__main__":
    demo = RoboticTaskDemo()
    demo.run()