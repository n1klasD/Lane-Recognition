import glob

import cv2 as cv

import config
from CVPipeline import PerspectiveTransform
from CVPipeline import Calibration


def main():
    # variables
    debug = True

    # camera calibration
    # get calibration images
    calibration_images = glob.glob(config.calibration_images_path + '/*.jpg')
    camera_calibration = Calibration()
    camera_calibration.calibrate(calibration_images)
    print("Calibration successful!")

    # apply to calibration to the source points that are used for the perspective transformation
    undistorted_points = camera_calibration.undistortPoints(config.sources_points)
    perspective_transform = PerspectiveTransform()
    perspective_transform.set_source_points(undistorted_points)



    # open video
    cap = cv.VideoCapture('resources/Udacity/project_video.mp4')
    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error opening video stream or file")

    # main loop
    # Read until video is completed
    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret:

            # debugging, show points of perspective transformation
            if config.PERSPECTIVE_DEBUG:
                for x, y in config.sources_points:
                    cv.circle(frame, (x, y), 2, (0, 0, 255), -1)

            cv.imshow('Normal video', frame)

            # transform frame here
            # -----------------------------------------

            frame = camera_calibration.undistort(frame, fix_roi=False)
            modified_frame = perspective_transform.transform(frame)

            # -----------------------------------------
            # don´t touch the code below

            if config.DEBUG:
                cv.imshow('Perspective Transformation', modified_frame)

            # normal video
            # Display the resulting frame
            cv.imshow('Camera Calibrated', frame)

            key = cv.waitKey(25)
            if debug:
                while key not in [ord('q'), ord('s'), ord('d')]:
                    key = cv.waitKey(10)
            # Quit when 'q' is pressed
            if key == ord('q'):
                break

            # show one frame at a time(s) in debug mode(toggle with d)
            if key == ord('d'):
                debug = not debug
        # Break the loop
        else:
            break
    # When everything done, release the video capture object
    cap.release()
    # Closes all the frames
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
