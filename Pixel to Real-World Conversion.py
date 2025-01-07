import cv2
import numpy as np

def load_calibration_data(calibration_file='calibration_data.npz'):
    """
    Load camera calibration data from a file.

    Args:
        calibration_file (str): Path to the calibration data file.

    Returns:
        camera_matrix (numpy.ndarray): Camera matrix (3x3).
        dist_coeffs (numpy.ndarray): Distortion coefficients.
    """
    calibration_data = np.load(calibration_file)
    camera_matrix = calibration_data['camera_matrix']
    dist_coeffs = calibration_data['dist_coeffs']
    return camera_matrix, dist_coeffs

def detect_checkerboard(image_path, checkerboard_size=(10, 7)):
    """
    Detect checkerboard corners in an image.

    Args:
        image_path (str): Path to the input image.
        checkerboard_size (tuple): Inner corners of the checkerboard (rows, cols).

    Returns:
        image (numpy.ndarray): The input image.
        corners_refined (numpy.ndarray): Refined corner locations.
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        return None, None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)

    if not ret:
        print("Checkerboard not detected in the image.")
        return image, None

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

    return image, corners_refined

def solve_pnp(objp, corners_refined, camera_matrix, dist_coeffs):
    """
    Solve the PnP problem to find rotation and translation vectors.

    Args:
        objp (numpy.ndarray): Object points in the real-world coordinate system.
        corners_refined (numpy.ndarray): Refined corner locations.
        camera_matrix (numpy.ndarray): Camera matrix (3x3).
        dist_coeffs (numpy.ndarray): Distortion coefficients.

    Returns:
        rvecs (numpy.ndarray): Rotation vector.
        tvecs (numpy.ndarray): Translation vector.
    """
    ret, rvecs, tvecs = cv2.solvePnP(objp, corners_refined, camera_matrix, dist_coeffs)
    if not ret:
        print("Failed to solve PnP.")
        return None, None
    return rvecs, tvecs

def pixel_to_real_world(pixel_coords, camera_matrix, rvecs, tvecs, dist_coeffs):
    """
    Convert pixel coordinates to real-world coordinates.

    Args:
        pixel_coords (tuple): (x, y) pixel coordinates.
        camera_matrix (numpy.ndarray): Camera matrix from calibration.
        rvecs (numpy.ndarray): Rotation vector from solvePnP.
        tvecs (numpy.ndarray): Translation vector from solvePnP.
        dist_coeffs (numpy.ndarray): Distortion coefficients.

    Returns:
        real_world_coords (numpy.ndarray): Real-world coordinates (X, Y, Z) in meters.
    """
    pixel_coords = np.array([[pixel_coords]], dtype=np.float32)
    undistorted_coords = cv2.undistortPoints(pixel_coords, camera_matrix, dist_coeffs)
    undistorted_coords = undistorted_coords.squeeze().reshape(2, 1)
    undistorted_coords = np.vstack((undistorted_coords, [1]))

    rotation_matrix, _ = cv2.Rodrigues(rvecs)
    inverse_rotation = np.linalg.inv(rotation_matrix)
    inverse_camera_matrix = np.linalg.inv(camera_matrix)
    tvecs = tvecs.reshape(3, 1)

    real_world_coords = inverse_rotation @ (inverse_camera_matrix @ undistorted_coords - tvecs)
    return real_world_coords.flatten()

def pixel_to_real_world_conversion(image_path, calibration_file='calibration_data.npz', checkerboard_size=(10, 7), square_size=0.025):
    """
    Perform pixel-to-real-world coordinate conversion using a checkerboard image.

    Args:
        image_path (str): Path to the input image.
        calibration_file (str): Path to the calibration data file.
        checkerboard_size (tuple): Inner corners of the checkerboard (rows, cols).
        square_size (float): Size of each square on the checkerboard in meters.

    Returns:
        real_world_coords (numpy.ndarray): Real-world coordinates (X, Y, Z) in meters.
    """
    # Load calibration data
    camera_matrix, dist_coeffs = load_calibration_data(calibration_file)

    # Detect checkerboard corners
    image, corners_refined = detect_checkerboard(image_path, checkerboard_size)
    if image is None or corners_refined is None:
        return None

    # Prepare object points in the real-world coordinate system
    objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2) * square_size

    # Solve PnP
    rvecs, tvecs = solve_pnp(objp, corners_refined, camera_matrix, dist_coeffs)
    if rvecs is None or tvecs is None:
        return None

    # Example: Convert a pixel coordinate to real-world coordinates
    pixel_x, pixel_y = corners_refined[0][0]  # Use the first detected corner as an example
    real_world_coords = pixel_to_real_world((pixel_x, pixel_y), camera_matrix, rvecs, tvecs, dist_coeffs)

    print(f"Pixel coordinates: ({pixel_x}, {pixel_y})")
    print(f"Real-world coordinates: {real_world_coords} meters")

    return real_world_coords