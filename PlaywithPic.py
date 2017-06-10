import cv2
import numpy as np
import copy
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

def count_mat(m,label):
    max_i,max_j = m.shape
    ans = []
    for i in range(label):
        ans.append(0)
    for i in range(0,max_i):
        bnc = (np.bincount(m[i][0:max_j])).tolist()
        if (len(bnc)<label):
            for i in range(label-len(bnc)):
                bnc.append(0)
        ans = add(ans,bnc)
    return ans

def add(ans,bnc):
    return [ans[i]+bnc[i] for i in range(len(ans))]

ori = cv2.imread('14.jpg',1)

# threshold value
'''
pic 1,4:    threshold 140
pic 14:     threshold 90
'''
t_val = 90
# structure element
kernel = np.ones((3,3),np.uint8)
ker_f_dila = np.ones((5,5),np.uint8)
# resize original image to 1/4
# ori = cv2.resize(ori,None,fx=0.25, fy=0.25, interpolation = cv2.INTER_CUBIC)
img = cv2.cvtColor(ori,cv2.COLOR_BGR2GRAY)
# use gamma correction to adjust black dot
ad = adjust_gamma(img,1.3)
# ad = adjust_gamma(ad,2)
# thresholding by using threshold value above
ret,thresh = cv2.threshold(img, t_val, 255, cv2.THRESH_BINARY)
# ret,thresh2 = cv2.threshold(img, t_val, 255, cv2.THRESH_BINARY_INV)
# dilation = cv2.dilate(thresh2,kernel,iterations = 1)
# ret,dilation = cv2.threshold(dilation, 127, 255, cv2.THRESH_BINARY_INV)
closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
ret,inv_open = cv2.threshold(opening, 127, 255, cv2.THRESH_BINARY_INV)
inv_open = cv2.dilate(inv_open,ker_f_dila,iterations = 1)
# ret,inv_open = cv2.threshold(inv_open, 127, 255, cv2.THRESH_BINARY_INV)

output = cv2.connectedComponentsWithStats(inv_open,8,cv2.CV_32S)
print('num_label',output[0])
print('label',(output[1]))

# cv2.imshow('threshold : '+str(t_val),thresh)
# cv2.imshow('inv_o',inv_open)
# cv2.imshow('closing',closing)
# cv2.imshow('opening',opening)
# cv2.imshow('ad',ad)

cnt = count_mat(output[1], output[0])
cnt2 =copy.copy(cnt)
cnt2.remove(max(cnt))
ind = []
ind.append(cnt.index(max(cnt)))
ind.append(cnt.index(max(cnt2)))

print('ind',ind)
print('stat',output[2])
print('centroid',output[3])
for i in output[3]:
    if (output[3].tolist()).index(i.tolist()) not in ind:
        cv2.circle(ori, (int(i[0]),int(i[1])), 20, (0, 0, 255), 2)

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
cv2.imshow('ori',ori)
# cv2.imshow('img',img)
# cv2.imshow('ad',ad)
# cv2.imshow('Laplacian',laplacian)
# cv2.imshow('threshold : '+str(t_val),thresh)
# cv2.imshow('inv_o',inv_open)
# cv2.imshow('closing',closing)
# cv2.imshow('opening',opening)
# cv2.imshow('sum',sum2)
cv2.waitKey(0)
cv2.destroyAllWindows()