import cv2
import numpy as np

class DetectingMarkers:
    def __init__(self):
        self.image_with_corners = None
        self.corners = None

    def detect(self, image_path):
        """
        Detect markers (edges and corners) in an image.
        :param image_path: Path to the input image.
        :return: True if markers are detected, False otherwise.
        """
        # Load the image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load image: {image_path}")
            return False

        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Edge Detection (Canny Edge Detector)
        edges = cv2.Canny(gray, threshold1=50, threshold2=150)  # Adjust thresholds as needed

        # Corner Detection (Shi-Tomasi Corner Detector)
        self.corners = cv2.goodFeaturesToTrack(gray, maxCorners=100, qualityLevel=0.01, minDistance=10)
        self.image_with_corners = image.copy()
        if self.corners is not None:
            self.corners = np.int32(self.corners)  # Use np.int32 instead of np.int0
            for corner in self.corners:
                x, y = corner.ravel()
                cv2.circle(self.image_with_corners, (x, y), 5, (0, 255, 0), -1)  # Mark corners in green

        # Display the results
        cv2.imshow('Original Image', image)
        cv2.imshow('Canny Edge Detection', edges)
        cv2.imshow('Shi-Tomasi Corner Detection', self.image_with_corners)

        # Wait for a key press and close windows
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return True

    def get_corners(self):
        """
        Get the detected corners.
        :return: List of corner coordinates.
        """
        return self.corners

if __name__ == "__main__":
    # Run the marker detection if this script is executed directly
    detector = DetectingMarkers()
    image_path = 'checkerboards/Checkerboard-A4-25mm-10x7.jpg'  # Replace with your image path
    detector.detect(image_path)