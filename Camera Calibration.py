import cv2
import numpy as np
import os

class CameraCalibration:
    def __init__(self, checkerboard_folder='checkerboards'):
        self.checkerboard_folder = checkerboard_folder
        self.camera_matrix = None
        self.dist_coeffs = None

    def calibrate(self):
        """
        Perform camera calibration using checkerboard images.
        :return: True if calibration is successful, False otherwise.
        """
        # List of checkerboard properties (inner corners)
        checkerboards = [
            {"name": "Checkerboard-A4-25mm-10x7.jpg", "vertices": (10, 7)},
            {"name": "Checkerboard-A3-25mm-15x10.jpg", "vertices": (15, 10)},
            {"name": "Checkerboard-A2-25mm-22x15.jpg", "vertices": (22, 15)},
            {"name": "Checkerboard-A1-25mm-32x22.jpg", "vertices": (32, 22)},
            {"name": "Checkerboard-A0-25mm-46x32.jpg", "vertices": (46, 32)},
        ]

        square_size = 0.025  # Each square is 25 mm = 0.025 meters

        # Arrays to store object points and image points
        obj_points = []  # 3D points in real-world space
        img_points = []  # 2D points in image plane

        for checkerboard in checkerboards:
            fname = os.path.join(self.checkerboard_folder, checkerboard["name"])
            img = cv2.imread(fname)
            if img is None:
                print(f"Failed to load image: {fname}")
                continue

            # Display the image to verify it is loaded correctly
            cv2.imshow('Image', img)
            cv2.waitKey(100)

            checkerboard_size = checkerboard["vertices"]
            objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
            objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2) * square_size

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find checkerboard corners
            ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)

            if ret:
                # Refine corner locations for better accuracy
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

                obj_points.append(objp)
                img_points.append(corners_refined)

                # Draw and display the detected corners
                cv2.drawChessboardCorners(img, checkerboard_size, corners_refined, ret)
                cv2.imshow('Detected Corners', img)
                cv2.waitKey(100)
            else:
                print(f"Checkerboard not detected in image: {fname}")

        cv2.destroyAllWindows()

        # Perform camera calibration
        if len(obj_points) > 0 and len(img_points) > 0:
            ret, self.camera_matrix, self.dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
                obj_points, img_points, gray.shape[::-1], None, None
            )

            if ret:
                print("Camera calibration successful!")
                print("\nCamera Matrix:\n", self.camera_matrix)
                print("\nDistortion Coefficients:\n", self.dist_coeffs)

                # Save calibration results to a file
                np.savez('calibration_data.npz', camera_matrix=self.camera_matrix, dist_coeffs=self.dist_coeffs)
                print("Calibration data saved to calibration_data.npz")
                return True
            else:
                print("Camera calibration failed.")
                return False
        else:
            print("Not enough valid calibration images were found.")
            return False

    def get_calibration_data(self):
        """
        Get the camera matrix and distortion coefficients.
        :return: camera_matrix, dist_coeffs (or None, None if calibration failed).
        """
        return self.camera_matrix, self.dist_coeffs

if __name__ == "__main__":
    # Run the calibration if this script is executed directly
    calibrator = CameraCalibration()
    calibrator.calibrate()