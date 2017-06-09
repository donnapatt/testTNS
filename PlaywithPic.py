import cv2
import numpy as np
import matplotlib.pyplot as plt

# function gamma correction
def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
    for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)

ori = cv2.imread('7.jpg',1)

# threshold value
t_val = 145
# structure element
kernel = np.ones((3,3),np.uint8)
# resize original image to 1/4
# img = cv2.resize(ori,None,fx=0.25, fy=0.25, interpolation = cv2.INTER_CUBIC)
img = cv2.cvtColor(ori,cv2.COLOR_BGR2GRAY)
# use gamma correction to adjust black dot
ad = adjust_gamma(img,1.3)
# ad = adjust_gamma(ad,2)
# thresholding by using threshold value above
ret,thresh = cv2.threshold(img, t_val, 255, cv2.THRESH_BINARY)
ret,thresh2 = cv2.threshold(img, t_val, 255, cv2.THRESH_BINARY_INV)
dilation = cv2.dilate(thresh2,kernel,iterations = 1)
ret,dilation = cv2.threshold(dilation, 127, 255, cv2.THRESH_BINARY_INV)
closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
# find edge by using laplacian filter
laplacian = cv2.Laplacian(opening,cv2.CV_64F)
# sum between laplacain and thresh
sum1 = laplacian+opening
sum2 = np.uint8(sum1)
sum2 = cv2.resize(sum2,None,fx=1.2, fy=1.2, interpolation = cv2.INTER_CUBIC)

# # left_top = (laplacian.shape[0]-1,laplacian.shape[1]-1)
# # right_top = (0,laplacian.shape[1]-1)
# # left_bottom = (0,laplacian.shape[1]-1)
# # right_bottom = (0,0)
# # for i in range(laplacian.shape[0]):
# #     for j in range(laplacian.shape[1]):
# #         if laplacian[i,j]==1:
# #             if(j>)

# print(laplacian.shape)
# for i in range
# show image
# plt.imshow(ori, cmap = 'gray')
# plt.show()
# cv2.imshow('ori',ori)
# cv2.imshow('img',img)
# cv2.imshow('ad',ad)
cv2.imshow('Laplacian',laplacian)
cv2.imshow('threshold : '+str(t_val),thresh)
cv2.imshow('dilation',dilation)
# cv2.imshow('closing',closing)
# cv2.imshow('opening',opening)
# cv2.imshow('sum',sum2)
cv2.waitKey(0)
cv2.destroyAllWindows()