import cv2
import numpy as np

class PixelToRealWorldConversion:
    def __init__(self, camera_matrix, dist_coeffs):
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs

    def convert(self, pixel_coords, rvecs, tvecs):
        """
        Convert pixel coordinates to real-world coordinates.
        :param pixel_coords: (x, y) pixel coordinates.
        :param rvecs: Rotation vector from solvePnP.
        :param tvecs: Translation vector from solvePnP.
        :return: Real-world coordinates (X, Y, Z) in meters.
        """
        # Convert pixel coordinates to a numpy array
        pixel_coords = np.array([[pixel_coords]], dtype=np.float32)

        # Undistort the pixel coordinates
        undistorted_coords = cv2.undistortPoints(pixel_coords, self.camera_matrix, self.dist_coeffs)

        # Reshape undistorted_coords to (2, 1)
        undistorted_coords = undistorted_coords.squeeze().reshape(2, 1)

        # Add a Z-coordinate (set to 1) to make it (3, 1)
        undistorted_coords = np.vstack((undistorted_coords, [1]))

        # Convert rotation vector to rotation matrix
        rotation_matrix, _ = cv2.Rodrigues(rvecs)

        # Compute the inverse of the rotation matrix and camera matrix
        inverse_rotation = np.linalg.inv(rotation_matrix)
        inverse_camera_matrix = np.linalg.inv(self.camera_matrix)

        # Reshape tvecs to (3, 1)
        tvecs = tvecs.reshape(3, 1)

        # Calculate the real-world coordinates
        real_world_coords = inverse_rotation @ (inverse_camera_matrix @ undistorted_coords - tvecs)
        return real_world_coords.flatten()

if __name__ == "__main__":
    # Example usage
    camera_matrix = np.array([[1000, 0, 320], [0, 1000, 240], [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
    converter = PixelToRealWorldConversion(camera_matrix, dist_coeffs)

    # Example pixel coordinates
    pixel_coords = (320, 240)
    rvecs = np.array([[0], [0], [0]], dtype=np.float32)
    tvecs = np.array([[0], [0], [0]], dtype=np.float32)

    real_world_coords = converter.convert(pixel_coords, rvecs, tvecs)
    print(f"Real-world coordinates: {real_world_coords} meters")