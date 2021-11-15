import numpy as np
import cv2 as cv

import config


def main():
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

            # Display the resulting frame
            cv.imshow('Frame', frame)

            # show one frame at a time in debug mode
            key = cv.waitKey(0)
            if config.DEBUG:
                while key not in [ord('q'), ord('s')]:
                    key = cv.waitKey(0)
            # Quit when 'q' is pressed
            if key == ord('q'):
                break
        # Break the loop
        else:
            break
    # When everything done, release the video capture object
    cap.release()
    # Closes all the frames
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
