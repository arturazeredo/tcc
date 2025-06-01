import numpy as np
import cv2
import gc

def image_resize(inputImg, max_=2560):
    resolution = [max_, max_]
    inputShape = inputImg.shape[:2]
    max_resolution = max(inputShape[0], inputShape[1])
    rad = 1
    if max_resolution > max_:
        rad = max_ / max_resolution

    resolution[0] = int(np.round(rad * inputShape[1]))
    resolution[1] = int(np.round(rad * inputShape[0]))

    img_resize = cv2.resize(inputImg, (resolution[0], resolution[1]))

    return img_resize



def rotate_bond(img, ang,back=True):
    (h, w) = img.shape[:2]
    (cX, cY) = (w//2,h//2)
    M = cv2.getRotationMatrix2D((cX,cY), -ang, 1.0)
    cos = np.abs(M[0,0])
    sin = np.abs(M[0,1])
    nw = int((h*sin) + (w*cos))
    nh = int((h*cos) + (w*sin))
    M[0,2] += (nw/2) - cX
    M[1,2] += (nh/2) - cY
    del cX, cY, cos, sin, w, h
    gc.collect()
    if back:
        return cv2.warpAffine(img, M, (nw,nh),borderValue=(255,255,255))
    else:
        return cv2.warpAffine(img, M, (nw,nh))




def softmax(x):
    return(np.exp(x)/np.exp(x).sum())