import cv2
import copy as copy
import matplotlib.pyplot as plt
import numpy as np

def multi_thresh(img,low,mid,fst,sec,thd):
    height,width = img.shape[:2]
    ret = copy.copy(img)
    for i in range(height):
        for j in range(width):
            if img[i][j] <= low:
                ret[i][j] = fst
            elif img[i][j]>low and img[i][j]<=mid:
                ret[i][j] = sec
            else:
                ret[i][j] = thd
    return ret
