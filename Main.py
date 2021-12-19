import glob
import cv2 as cv
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

    optimizer = CVPipeline.Optimizer(1.4, max_cached_frames=15)
    measurement = CVPipeline.Measurement(target_time=50)

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
    START_POINT = 0  # first critical part
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
    curve_fitter = CurveFitter()

    while cap.isOpened():
        measurement.beginFrame()

        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret:

            # debugging, show points of perspective transformation
            if config.PERSPECTIVE_DEBUG:
                for x, y in config.sources_points:
                    cv.circle(frame, (x, y), 2, (0, 0, 255), -1)
            # transform frame here
            # -----------------------------------------

            # apply the camera calibration
            if config.ACTIVATE_CAMERA_CALIBRATION:
                modified_frame = camera_calibration.undistort(frame, fix_roi=False)
                final_frame = modified_frame.copy()
            else:
                modified_frame = frame

            # blurring
            modified_frame = Pipeline.gaussian_blur(modified_frame)
            
            # crop a ROI from the image
            modified_frame = roi.apply_roi(modified_frame, do_crop=False)

            # apply the perspective transform
            modified_frame = perspective_transform.transform(modified_frame)

            lane_extraction_timing = measurement.measure("Lane Extraction")
            white_yellow = Pipeline.extract_lanes(modified_frame)
            lane_extraction_timing.finish()
            cv.imshow("fefe", white_yellow)

            needs_update, diff_value = optimizer.needs_update(white_yellow)

            # curve fitting
            curve_timing = measurement.measure('Curve Fitting')
            #left, right = Pipeline.split_left_right(white_yellow)
            #y1, x1, _, _, _ = CurveFitter.fit_curve_polyfit(right)
            #y2, x2, _, _, _ = CurveFitter.fit_curve_polyfit(left)
            curve_timing.finish()

            final_frame = curve_fitter.calculateOverlay(white_yellow, perspective_transform, final_frame)

            # Inverse transform points and draw poly on original image
            inverse_timing = measurement.measure("Inverse Transformation")
            #points = CurveFitter.stack_points(y1, x1, y2, x2)
            #inv_points = perspective_transform.inverse_transform(points)
            #final_frame = CurveFitter.draw_area(final_frame, inv_points)
            inverse_timing.finish()

            # ---------- Transform the %resulting images perspective ----------- #

            measurement.endFrame()

            measurement.drawFrameTiming(final_frame)
            measurement.drawText(final_frame, 'Diff: {:.3f}'.format(diff_value), 2)
            measurement.drawText(final_frame, 'Threshold: {:.3f}'.format(optimizer.threshold), 3)
            measurement.drawText(final_frame, 'Max Cached: {}'.format(optimizer.max_cached_frames), 4)

            if needs_update:
                measurement.drawText(final_frame, 'Update', 5)

            measurement.drawTiming(final_frame, lane_extraction_timing, 2)
            measurement.drawTiming(final_frame, inverse_timing, 1)
            measurement.drawTiming(final_frame, curve_timing, 0)

            cv.imshow('Lane Detection', final_frame)

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
