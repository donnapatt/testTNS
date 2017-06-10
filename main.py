import cv2
import Detection as dt
import numpy as np


'''
case black dot
pic 1,7:    threshold 140
pic 5:      threshold 173
pic 14:     threshold 90
pic 18:     threshold 220
pic 19:     threshold 91
#############################################
case white_dot
pic 8:
thresh = multi_thresh(img,140,240,0,255,0)
kernel = np.ones((3,3),np.uint8)
ker_f_dila = np.ones((3,3),np.uint8)
#############################################
'''
ori = cv2.imread('8.jpg',1)
'''
threshold value
'''
t_val = 91

'''
resize original image to 1/4
'''
# ori = cv2.resize(ori,None,fx=0.25, fy=0.25, interpolation = cv2.INTER_CUBIC)
img = cv2.cvtColor(ori,cv2.COLOR_BGR2GRAY)
'''
use gamma correction to adjust black dot
'''
ad = dt.adjust_gamma(img,1.3)
# ad = adjust_gamma(ad,2)
'''
structure element
'''
# # for black
# kernel = np.ones((2,2),np.uint8)
# ker_f_dila = np.ones((5,5),np.uint8)
# for white
kernel = np.ones((3,3),np.uint8)
ker_f_dila = np.ones((3,3),np.uint8)
'''
thresholding by using threshold value above
'''
# ret,thresh = cv2.threshold(img, t_val, 255, cv2.THRESH_BINARY)
thresh = dt.multi_thresh(img,140,240,0,255,0)
closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
ret,inv_open = cv2.threshold(opening, 127, 255, cv2.THRESH_BINARY_INV)
inv_open = cv2.dilate(inv_open,ker_f_dila,iterations = 1)
'''
write circle
'''
output = cv2.connectedComponentsWithStats(inv_open,8,cv2.CV_32S)
dt.write_circle(output,ori)
'''
show image
'''
# plt.imshow(img, cmap = 'gray')
# plt.show()
cv2.imshow('ori',ori)
# cv2.imshow('img',img)
# cv2.imshow('ad',ad)
cv2.imshow('threshold : '+str(t_val),thresh)
cv2.imshow('inv_o',inv_open)
cv2.imshow('closing',closing)
cv2.imshow('opening',opening)
cv2.waitKey(0)
cv2.destroyAllWindows()