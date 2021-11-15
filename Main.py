import cv2 as cv

import config
from CVPipeline import PerspectiveTransform


def main():
    # helpers
    test = PerspectiveTransform()

    # variables
    debug = False

    # main loop

    cap = cv.VideoCapture('resources/Udacity/project_video.mp4')
    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error opening video stream or file")
    # Read until video is completed
    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret:

            # transform frame here
            mod_frame = test.transform(frame)

            if config.DEBUG:
                cv.imshow('Lane Recognition', mod_frame)

            # Display the resulting frame
            cv.imshow('Normal video', frame)

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
