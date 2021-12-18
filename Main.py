import glob
import cv2 as cv
import numpy as np
from CVPipeline.CurveFitter import CurveFitter
import config
from CVPipeline import PerspectiveTransform
from CVPipeline import Calibration
from CVPipeline import ROI
from CVPipeline import Pipeline
def main():
    # variables
    debug = False
    perspective_transform = PerspectiveTransform()
    camera_calibration = Calibration()
    # camera calibration
    # get calibration images
    if config.ACTIVATE_CAMERA_CALIBRATION:
        calibration_images = glob.glob(config.calibration_images_path + '/*.jpg')
        camera_calibration.calibrate(calibration_images)
        # apply calibration to the source points that are used for the perspective transformation
        undistorted_points = camera_calibration.undistortPoints(config.sources_points)
        #
        perspective_transform.set_source_points(undistorted_points)
    else:
        perspective_transform.set_source_points(config.sources_points)
    # open video
    cap = cv.VideoCapture('resources/Udacity/project_video.mp4')
    # configure camera
    cap.set(cv.CAP_PROP_FPS, 240)
    # start video playback at critical point, where street changes color
    # START_POINT = 0 # Start
    START_POINT = 500  # first critical part
    cap.set(cv.CAP_PROP_POS_FRAMES, START_POINT)
    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error opening video stream or file")
    video_width = cap.get(cv.CAP_PROP_FRAME_WIDTH)
    video_height = cap.get(cv.CAP_PROP_FRAME_HEIGHT)
    # initialize ROI
    # roi = ROI(video_width, video_height, distance_to_middle=90, bottom_dist_side=30, top_dist_side=420)
    roi = ROI()
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
            # cv.imshow('Normal video', roi.draw_roi(frame))
            # transform frame here
            # -----------------------------------------
            # blurring
            modified_frame = Pipeline.gaussian_blur(frame)
            # crop a ROI from the image
            modified_frame = roi.apply_roi(modified_frame)
            # cv.imshow("ROI", modified_frame)
            # todo: maybe also crop video
            # apply the camera calibration
            if config.ACTIVATE_CAMERA_CALIBRATION:
                modified_frame = camera_calibration.undistort(modified_frame, fix_roi=False)
            # segment yellow lane
            yellow_lane = Pipeline.extract_yellow_lane(modified_frame)
            # cv.imshow("Yellow lane", yellow_lane)
            # segment white lane
            white_lane = Pipeline.extract_white_lane(modified_frame)
            # cv.imshow("White Lane", white_lane)

            # make canny edge detection and apply the roi to it
            canny = Pipeline.canny_edge_detection(frame)
            canny = roi.apply_roi(canny)
            # cv.imshow("Canny", canny)

            # combine white and yellow lane
            white_yellow = cv.bitwise_or(yellow_lane, white_lane)
            white_yellow = cv.cvtColor(white_yellow, cv.COLOR_RGB2GRAY)
            cv.imshow("white_yellow", white_yellow)

            # combine canny and white/yellow
            modified_frame = cv.bitwise_or(canny, white_yellow)
            # modified_frame = cv.bitwise_or(canny, white_yellow)

            # make canny edge detection and apply the roi to it
            canny = Pipeline.canny_edge_detection(white_yellow)
            # canny = roi.apply_roi(canny)
            cv.imshow("Canny", canny)

            # print(modified_frame.shape)

            # curve transformation
            left, right = Pipeline.split_left_right(white_yellow)
            cv.imshow("left", left)
            right, left = Pipeline.split_left_right(canny)
            # cv.imshow("left", left)
            cv.imshow("right", right)
            x1, y1 = CurveFitter.fit_curve_polyfit(left)
            x2, y2 = CurveFitter.fit_curve_polyfit(right)
            frame = CurveFitter.draw_area(frame,x1,y1,x2,y2)
            frame[x1,y1] = (0,0,255)
            frame[x2,y2] = (0,0,255)
            
            cv.imshow("curved", frame)
            # ---------- Transform the %resulting images perspective ----------- #
            cv.imshow('Lane Detection', modified_frame)
            # apply the perspective transform
            frame = perspective_transform.transform(frame)
            # cv.imshow('perspective transform', frame)
            # -----------------------------------------
            # don´t touch the code below
            # normal video
            # Display the resulting frame
            key = cv.waitKey(1)
            if debug:
                while key not in [ord('q'), ord('s'), ord('d')]:
                    key = cv.waitKey(1)
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