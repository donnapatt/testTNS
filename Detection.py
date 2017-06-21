import cv2
import numpy as np
import copy

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

def write_circle(output,ori):
    cnt = count_mat(output[1], output[0])
    cnt2 = copy.copy(cnt)
    cnt2.remove(max(cnt))
    ind = []
    ind.append(cnt.index(max(cnt)))
    ind.append(cnt.index(max(cnt2)))

    print('ind', ind)
    print('stat', output[2])
    print('centroid', output[3])
    for i in output[3]:
        if (output[3].tolist()).index(i.tolist()) not in ind:
            cv2.circle(ori, (int(i[0]), int(i[1])), 20, (255, 255, 255), 2)

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