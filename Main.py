import glob
import cv2 as cv
import numpy as np
from CVPipeline.CurveFitter import CurveFitter
import config
import CVPipeline
from CVPipeline import PerspectiveTransform
from CVPipeline import Calibration
from CVPipeline import ROI
from CVPipeline import Pipeline


def main():
    # variables
    debug = False
    perspective_transform = PerspectiveTransform()
    camera_calibration = Calibration()

    optimizer = CVPipeline.Optimizer(0.8, max_cached_frames=15)
    measurement = CVPipeline.Measurement(target_time=50)

    white_timing = measurement.measure('White Lane')
    curve_timing = measurement.measure('Curve Fitting')
    yellow_timing = measurement.measure('Yellow Lane')
    canny_timing = measurement.measure('Canny')

    x1 = y1 = x2 = y2 = 0

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
    # start video playback at critical point, where street changes color
    # START_POINT = 0 # Start
    START_POINT = 700  # first critical part
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
        measurement.beginFrame()

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
            modified_frame = roi.apply_roi(modified_frame, do_crop=True)
            # cv.imshow("ROI", modified_frame)
            # todo: maybe also crop video
            # apply the camera calibration
            if config.ACTIVATE_CAMERA_CALIBRATION:
                modified_frame = camera_calibration.undistort(modified_frame, fix_roi=False)

            update_timing = measurement.measure('check_update')
            gray_roi = cv.cvtColor(modified_frame, cv.COLOR_RGB2GRAY)
            _, gray_roi = cv.threshold(gray_roi, thresh=130, maxval=255, type=cv.THRESH_BINARY)
            needs_update, diff_value = optimizer.needs_update(gray_roi)
            update_timing.finish()

            optimizer_debug_frame = optimizer.draw_debug_img(gray_roi)
            cv.imshow('Optimizer Diff', optimizer_debug_frame)

            if needs_update:
                # segment white lane
                white_timing.reset()
                white_lane = Pipeline.extract_white_lane(modified_frame)
                white_timing.finish()
                cv.imshow("White Lane", white_lane)

                # segment yellow lane
                yellow_timing.reset()
                yellow_lane = Pipeline.extract_yellow_lane(modified_frame)
                yellow_timing.finish()
                cv.imshow("Yellow Lane", yellow_lane)

                # combine white and yellow lane
                white_yellow = cv.bitwise_or(yellow_lane, white_lane)
                white_yellow = cv.cvtColor(white_yellow, cv.COLOR_RGB2GRAY)
                # cv.imshow("white_yellow", white_yellow)

                # make canny edge detection and apply the roi to it
                canny_timing.reset()
                canny = Pipeline.canny_edge_detection(white_yellow)
                canny_timing.finish()
                cv.imshow("Canny", canny)

                # dilate canny
                # dilated_canny = Pipeline.dilate(canny)
                # cv.imshow("Dilated Canny", dilated_canny)

                # curve transformation
                curve_timing.reset()
                left, right = Pipeline.split_left_right(canny)
                # cv.imshow("left", left)
                # cv.imshow("right", right)
                x1, y1 = CurveFitter.fit_curve_polyfit(left)
                x2, y2 = CurveFitter.fit_curve_polyfit(right)
                # frame[x1, y1] = (0, 0, 255)
                # frame[x2, y2] = (0, 0, 255)
                curve_timing.finish()

            draw_timing = measurement.measure('draw_area')
            area = CurveFitter.poly_area(modified_frame, x1, y1, x2, y2)
            area = roi.reverse(area)
            frame = cv.addWeighted(frame, 1, area, 0.3, 0)
            draw_timing.finish()

            # cv.imshow("curved", frame)
            # ---------- Transform the %resulting images perspective ----------- #
            # cv.imshow('Lane Detection', modified_frame)

            measurement.endFrame()
            final_frame = frame.copy()

            measurement.drawFrameTiming(final_frame)
            measurement.drawText(final_frame, 'Diff: {:.3f}'.format(diff_value), 2)
            measurement.drawText(final_frame, 'Threshold: {:.3f}'.format(optimizer.threshold), 3)
            measurement.drawText(final_frame, 'Max Cached: {}'.format(optimizer.max_cached_frames), 4)

            if needs_update:
                measurement.drawText(final_frame, 'Update', 5)

            measurement.drawTiming(final_frame, update_timing, 4)
            measurement.drawTiming(final_frame, white_timing, 3)
            measurement.drawTiming(final_frame, yellow_timing, 2)
            measurement.drawTiming(final_frame, canny_timing, 1)
            measurement.drawTiming(final_frame, curve_timing, 0)
            measurement.drawTiming(final_frame, draw_timing, 5)

            cv.imshow('Final', final_frame)

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

            if key == ord('u'):
                optimizer.threshold -= 0.1
            elif key == ord('i'):
                optimizer.threshold += 0.1
        # Break the loop
        else:
            break
    # When everything done, release the video capture object
    cap.release()
    # Closes all the frames
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
