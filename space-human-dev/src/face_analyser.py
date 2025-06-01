from src.utils import image_resize
import cv2
import numpy as np


class Face:
    def __init__(self,size=480):
        self.path_face = "src/models/face.xml"
        self.haar_face = cv2.CascadeClassifier(self.path_face)
        self.path_eyesr = "src/models/right.xml"
        self.haar_eyes_r = cv2.CascadeClassifier(self.path_eyesr)
        self.path_eyesl = "src/models/left.xml"
        self.haar_eyes_l = cv2.CascadeClassifier(self.path_eyesl)
        self.size = size

    def preprocess(self,inputImg):
        #img_r = image_resize(inputImg,max_=self.size)
        gray = cv2.cvtColor(inputImg,cv2.COLOR_BGR2GRAY)
        return None, gray

    def face_box(self,inputImg,scaleF=1.05,minN=5,minSize=(30,30)):
        face_react = []
        if inputImg is not None:
            face_react = self.haar_face.detectMultiScale(inputImg, scaleFactor=scaleF,
                                                         minNeighbors=minN, minSize=minSize,
                                                         flags=cv2.CASCADE_SCALE_IMAGE)
        return face_react

    def eyes_box_r(self,inputImg,scaleF=1.1,minN=10,minSize=(15,15)):
        eyes_r_react = []
        if inputImg is not None:
            eyes_r_react = self.haar_eyes_r.detectMultiScale(inputImg, scaleFactor=scaleF,
                                                           minNeighbors=minN, minSize=minSize,
                                                           flags=cv2.CASCADE_SCALE_IMAGE)
        return eyes_r_react

    def eyes_box_l(self, inputImg, scaleF=1.1, minN=10, minSize=(15, 15)):
        eyes_l_react = []
        if inputImg is not None:
            eyes_l_react = self.haar_eyes_l.detectMultiScale(inputImg, scaleFactor=scaleF,
                                                           minNeighbors=minN, minSize=minSize,
                                                           flags=cv2.CASCADE_SCALE_IMAGE)

        return eyes_l_react

    def cut_face(self,image_pre,box,index=0):
        try:
            x, y, w, h = box[index]
            cut = image_pre[y:y+h,x:x+w]
        except:
            return None
        return cut

    def face_extract(self, inputImg):
        img_r, image = self.preprocess(inputImg)
        face_B = self.face_box(image)

        if len(face_B) > 0:
            image_face = self.cut_face(image, face_B, index=0)
            eyes_R = self.eyes_box_l(image_face)
            eyes_L = self.eyes_box_r(image_face)
            return img_r, face_B, eyes_L, eyes_R
        else:
            return img_r, [], [], []