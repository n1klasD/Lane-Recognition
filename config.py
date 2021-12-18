"""
Script for Configuration Variables
"""

# Enter debug mode
DEBUG = True

# Perspective Transformation

# Start debug mode
PERSPECTIVE_DEBUG = False
# corresponding source and destination points for the perspective transformation (must be 4)
sources_points = [[279, 688], [1026, 688], [684, 450], [596, 450]]
destination_points = [[300, 720], [980, 720], [980, 0], [300, 0]]

# Camera Calibration

# Start debug mode
ACTIVATE_CAMERA_CALIBRATION = False
CALIBRATION_DEBUG = False
# Path to camera calibration images (folder, that must contain jpg images)
calibration_images_path = 'resources/Udacity/calib'

# ROI Points
roi_points = [[],[],[],[]]



