"""

"""
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


class CurveFitter:

    @staticmethod
    def fit_curve_polyfit(img):
        new_frame = img.copy()  
        w = np.where(new_frame != 0)
        x = w[0]
        y = w[1]

        xn = np.arange(1270 - 1)  # x-Koordinaten

        ar = np.poly1d(np.polyfit(y, x, 2))
        y = np.int32(ar(xn))
        mask = (y >= 0) & (y >= 450) & (y < 720)
        # [( int(y[i]),x) for i,x in enumerate(xn) if y[i] >=0 and y[i] <= 720]
        return y[mask], xn[mask]


    @staticmethod
    def draw_area(frame,x1,y1,x2,y2):
        
        new_frame = frame.copy()
        
      #  n_empty1 = np.zeros(shape=new_frame.shape)
      #  n_empty2= np.zeros(shape=new_frame.shape)
      #  n_empty1[x1,y1] = 255
      #  n_empty2[x2,y2] = 255
      #  A = np.column_stack([x1,y1])
      #  B = np.column_stack([x2,y2])
        A = np.stack((y1,x1),axis=-1)
        B = np.stack((y2,x2),axis=-1)
        C= np.concatenate((A,B))
        print(C, C.shape)

        cv.fillPoly(new_frame, [C.astype(np.int32)], (0,255, 0))
        return new_frame

