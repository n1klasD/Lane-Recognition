import numpy as np
import cv2 as cv
import glob

DEBUG = False


class CameraCalibration:
    def __init__(self):
        # configuration of grid size
        self.grid_x = 6
        self.grid_y = 9

        # termination criteria
        self.criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # Arrays to store object points and image points from all the images.
        self.objpoints = []  # 3d point in real world space
        self.imgpoints = []  #

        # todo: out
        self.images = glob.glob('resources/Udacity/calib/*.jpg')

    def calibrate(self, image_paths):
        images = image_paths

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(grid_x,5,0)
        objp = np.zeros((self.grid_x * self.grid_y, 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.grid_y, 0:self.grid_x].T.reshape(-1, 2)

        for fname in self.images:
            if DEBUG:
                print(f"Evaluating: {fname}")

            img = cv.imread(fname, cv.IMREAD_GRAYSCALE)
            # Find the chess board corners
            ret, corners = cv.findChessboardCorners(img, (self.grid_y, self.grid_x), None)

            # If found, add object points, image points (after refining them)
            if ret:
                self.objpoints.append(objp)

                corners2 = cv.cornerSubPix(img, corners, (11, 11), (-1, -1), self.criteria)
                self.imgpoints.append(corners)

                # Draw and display the corners
                if DEBUG:
                    cv.drawChessboardCorners(img, (self.grid_y, self.grid_x), corners2, ret)
                    cv.imshow('img', img)
                    cv.waitKey(10)
                    print("\t--> good")
            elif DEBUG:
                print("\t--> faulty")

print(len(imgpoints))

# calibration
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, img.shape[::-1], None, None)

# Undistortion
img = cv.imread('resources/Udacity/calib/calibration3.jpg')
h, w = img.shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

# undistort
dst = cv.undistort(img, mtx, dist, None, newcameramtx)

# cropping
x, y, w, h = roi
dst = dst[y:y + h, x:x + w]

# show tests
cv.imshow("distorted", img)
cv.waitKey(5000)

# show tests
cv.imshow("Undistort", dst)
cv.waitKey(5000)

cv.destroyAllWindows()

# print(imgpoints)
print(len(imgpoints))
