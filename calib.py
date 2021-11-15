import numpy as np
import cv2 as cv
import glob

grid_x = 6
grid_y = 7

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(grid_x,5,0)
objp = np.zeros((grid_x * grid_y, 3), np.float32)
objp[:, :2] = np.mgrid[0:grid_y, 0:grid_x].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.
images = glob.glob('resources/Udacity/calib/*.jpg')

for fname in images:
    print(f"Evaluating: {fname}")

    img = cv.imread(fname, cv.IMREAD_GRAYSCALE)
    thresh_img = cv.threshold(img, thresh=10, maxval=255, type=cv.THRESH_BINARY)
    cv.imshow('testt', thresh_img)
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(np.float32(thresh_img), (grid_y, grid_x), None)

    # If found, add object points, image points (after refining them)
    if ret:
        objpoints.append(objp)

        corners2 = cv.cornerSubPix(thresh_img, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners)

        # Draw and display the corners
        cv.drawChessboardCorners(img, (grid_y, grid_x), corners2, ret)
        cv.imshow('img', img)
        cv.waitKey(10)
        print("\t--> good")
    else:
        print("\t--> faulty")

cv.destroyAllWindows()

# print(imgpoints)
print(len(imgpoints))
