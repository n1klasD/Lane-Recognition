"""

"""

import numpy as np
import cv2 as cv

import config


class Calibration:
    def __init__(self):
        # configuration of grid size
        self.grid_x = 6
        self.grid_y = 9

        self.mtx = None
        self.dist = None
        self.h = None
        self.w = None
        self.newcameramtx = None
        self.roi = None

    def calibrate(self, image_paths):
        images = image_paths

        # termination criteria
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # Arrays to store object points and image points from all the images.
        objpoints = []  # 3d point in real world space
        imgpoints = []  #

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(grid_x,5,0)
        objp = np.zeros((self.grid_x * self.grid_y, 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.grid_y, 0:self.grid_x].T.reshape(-1, 2)

        img = None

        for fname in images:
            if config.CALIBRATION_DEBUG:
                print(f"Evaluating: {fname}")

            img = cv.imread(fname, cv.IMREAD_GRAYSCALE)
            # Find the chess board corners
            ret, corners = cv.findChessboardCorners(img, (self.grid_y, self.grid_x), None)

            # If found, add object points, image points (after refining them)
            if ret:
                objpoints.append(objp)

                corners2 = cv.cornerSubPix(img, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners)

                # Draw and display the corners
                if config.CALIBRATION_DEBUG:
                    cv.drawChessboardCorners(img, (self.grid_y, self.grid_x), corners2, ret)
                    cv.imshow('img', img)
                    cv.waitKey(10)
                    print("\t--> good")
            elif config.CALIBRATION_DEBUG:
                print("\t--> faulty")

        # calibration
        ret, self.mtx, self.dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, img.shape[::-1], None, None)

        # Undistortion
        self.h, self.w = img.shape[:2]
        self.newcameramtx, self.roi = cv.getOptimalNewCameraMatrix(self.mtx, self.dist, (self.w, self.h), 0,
                                                                   (self.w, self.h))
        if config.CALIBRATION_DEBUG:
            cv.destroyAllWindows()

    def undistort(self, img, fix_roi=False):

        try:
            # undistort
            undistorted_image = cv.undistort(img, self.mtx, self.dist, None, self.newcameramtx)
        except:
            raise Exception("Call calibrate() before undistorting the image")

        if fix_roi:
            # cropping
            x, y, w, h = self.roi
            roi_extraction = undistorted_image[y:y + h, x:x + w]
            return roi_extraction
        else:
            return undistorted_image

    def undistortPoints(self, points):
        # expects a list of lists [x,y]

        np_points = np.array([points], dtype=np.float32)
        return cv.undistortPoints(np_points, self.mtx, self.dist, None, self.newcameramtx)

    def getCalibrationMatrix(self):
        return self.newcameramtx
