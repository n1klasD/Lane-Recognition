"""
Script for Configuration Variables
"""
# Image properties
VIDEO_HEIGHT = 720
VIDEO_WIDTH = 1280

# Enter debug mode
DEBUG = True

# Perspective Transformation

# Start debug mode
PERSPECTIVE_DEBUG = False
# corresponding source and destination points for the perspective transformation (must be 4)
sources_points = [[278, 688], [1026, 688], [684, 448], [598, 448]]
destination_points = [[300, 720], [980, 720], [980, 0], [300, 0]]

# Camera Calibration

# Start debug mode
ACTIVATE_CAMERA_CALIBRATION = True
CALIBRATION_DEBUG = False
# Path to camera calibration images (folder, that must contain jpg images)
calibration_images_path = 'resources/Udacity/calib'

# ROI Points
outer_roi_points = [[110, 665], [1230, 665], [780, 440], [545, 440]]
inner_roi_points = [[670, 470], [440, 665], [930, 665]]

# Parameters for curve fitting optimization
FRAME_LIFETIME = 20
TOLERANCE = 0.5
