from src.utils import image_resize
import numpy as np
import dlib
import cv2

class LandMask:
    def __init__(self,size=480):
        self.path = "src/models/face_landmarks.dat"
        self.detector = dlib.get_frontal_face_detector()
        self.model = dlib.shape_predictor(self.path)
        self.size = size

    def preprocess(self,inputImg):
        img_r = image_resize(inputImg,max_=self.size)
        gray = cv2.cvtColor(img_r,cv2.COLOR_BGR2GRAY)
        return img_r, gray

    def rect_to_bb(self,rect):
        x = rect.left()
        y = rect.top()
        w = rect.right() - x
        h = rect.bottom() - y
        return (x, y, w, h)

    def shape_to_np(self, shape, dtype="int"):
        coords = np.zeros((shape.num_parts, 2), dtype=dtype)

        for i in range(0, shape.num_parts):
            coords[i] = (shape.part(i).x, shape.part(i).y)

        return coords

    def extract(self,inputImg,index=0):

        img_r, gray = self.preprocess(inputImg)
        rects = self.detector(gray, 1)

        if len(rects) > 0:
            shape = self.model(gray, rects[index])
            points = self.shape_to_np(shape)
            face_box = self.rect_to_bb(rects[index])

            return img_r, points, face_box
        else:
            return img_r, [], []