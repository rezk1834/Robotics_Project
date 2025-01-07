import cv2
import numpy as np

def detect_markers(image_path, canny_threshold1=50, canny_threshold2=150, shi_tomasi_max_corners=100, shi_tomasi_quality_level=0.01, shi_tomasi_min_distance=10):
    """
    Detect edges and corners in an image using Canny edge detection and Shi-Tomasi corner detection.

    Args:
        image_path (str): Path to the input image.
        canny_threshold1 (int): First threshold for Canny edge detection.
        canny_threshold2 (int): Second threshold for Canny edge detection.
        shi_tomasi_max_corners (int): Maximum number of corners to detect using Shi-Tomasi.
        shi_tomasi_quality_level (float): Quality level for Shi-Tomasi corner detection.
        shi_tomasi_min_distance (int): Minimum distance between detected corners.

    Returns:
        edges (numpy.ndarray): Image with edges detected using Canny.
        image_shi_tomasi (numpy.ndarray): Image with corners detected using Shi-Tomasi.
    """
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        return None, None

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Edge Detection (Canny Edge Detector)
    edges = cv2.Canny(gray, threshold1=canny_threshold1, threshold2=canny_threshold2)

    # Corner Detection (Shi-Tomasi Corner Detector)
    shi_tomasi_corners = cv2.goodFeaturesToTrack(gray, maxCorners=shi_tomasi_max_corners, qualityLevel=shi_tomasi_quality_level, minDistance=shi_tomasi_min_distance)
    image_shi_tomasi = image.copy()
    if shi_tomasi_corners is not None:
        shi_tomasi_corners = np.int32(shi_tomasi_corners)  # Use np.int32 instead of np.int0
        for corner in shi_tomasi_corners:
            x, y = corner.ravel()
            cv2.circle(image_shi_tomasi, (x, y), 5, (0, 255, 0), -1)  # Mark corners in green

    return edges, image_shi_tomasi