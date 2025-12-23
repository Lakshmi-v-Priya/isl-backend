import cv2
import numpy as np

def is_blurry(frame, threshold=60):
    """
    Detect camera blur using Laplacian variance
    Lower threshold for deployed environments
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance < threshold


def is_low_light(frame, threshold=25):
    """
    Detect low light based on average brightness
    Lower threshold for browser cameras
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)
    return brightness < threshold
