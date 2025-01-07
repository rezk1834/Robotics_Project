import cv2
import numpy as np
import pybullet as p
import pybullet_data
import time

# Import functions from the three modules
from Camera_Calibration import calibrate_camera
from Detecting_Markers import detect_markers
from Pixel_to_Real_World_Conversion import pixel_to_real_world_conversion

# Initialize PyBullet
physicsClient = p.connect(p.GUI)  # Connect to the PyBullet GUI
p.setAdditionalSearchPath(pybullet_data.getDataPath())  # Set the data path for PyBullet assets
p.setGravity(0, 0, -9.81)  # Set gravity

# Load the plane and robot
planeId = p.loadURDF("plane.urdf")  # Load a plane
robotId = p.loadURDF("r2d2.urdf", [0, 0, 1])  # Load a robot (e.g., R2D2)

# Function to move the robot to a target position
def move_robot_to_target(robotId, target_position):
    """
    Move the robot to a target position.
    :param robotId: ID of the robot in PyBullet.
    :param target_position: Target position (X, Y, Z) in meters.
    """
    # Set the target position for the robot
    p.resetBasePositionAndOrientation(robotId, target_position, [0, 0, 0, 1])

# Function to simulate picking up an object
def pick_up_object(robotId, objectId):
    """
    Simulate picking up an object.
    :param robotId: ID of the robot in PyBullet.
    :param objectId: ID of the object in PyBullet.
    """
    # Move the robot to the object's position
    object_position, _ = p.getBasePositionAndOrientation(objectId)
    move_robot_to_target(robotId, object_position)

    # Simulate picking up the object (attach the object to the robot)
    p.createConstraint(robotId, -1, objectId, -1, p.JOINT_FIXED, [0, 0, 0], [0, 0, 0], object_position)

# Main function
def main():
    # Step 1: Perform Camera Calibration
    print("Running Camera Calibration...")
    camera_matrix, dist_coeffs = calibrate_camera()

    if camera_matrix is None or dist_coeffs is None:
        print("Camera calibration failed. Exiting.")
        return

    # Step 2: Detect Markers in an Image
    print("\nRunning Marker Detection...")
    image_path = "checkerboards/Checkerboard-A4-25mm-10x7.jpg"  # Replace with your image path
    image, corners_refined = detect_markers(image_path)

    if corners_refined is None:
        print("Marker detection failed. Exiting.")
        return

    # Step 3: Convert Pixel Coordinates to Real-World Coordinates
    print("\nRunning Pixel to Real-World Conversion...")
    real_world_coords = pixel_to_real_world_conversion(image_path)

    if real_world_coords is None:
        print("Pixel-to-real-world conversion failed. Exiting.")
        return

    print(f"Real-world coordinates: {real_world_coords} meters")

    # Step 4: Simulate Robotic Task
    print("\nSimulating Robotic Task...")

    # Load an object to pick up
    objectId = p.loadURDF("cube_small.urdf", [1, 1, 1])  # Load a small cube as the object

    # Move the robot to the real-world coordinates
    move_robot_to_target(robotId, real_world_coords)

    # Simulate picking up the object
    pick_up_object(robotId, objectId)

    # Run the simulation for a few seconds
    for _ in range(1000):
        p.stepSimulation()
        time.sleep(1.0 / 240.0)  # Simulate at 240 Hz

    # Disconnect from PyBullet
    p.disconnect()

if __name__ == "__main__":
    main()