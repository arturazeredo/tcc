import cv2
import numpy as np
import onnxruntime as onnx
from src.face_analyser import Face
import json

class FaceK:
    def __init__(self):
        self.path = "src/models/anti.onnx"
        self.sess = onnx.InferenceSession(self.path)
        self.inputs = [n.name for n in self.sess.get_inputs()]
        self.path_face = "src/models/face.xml"
        self.haar_face = cv2.CascadeClassifier(self.path_face)


    def cut_face(self,image_pre,box,index=0):
        try:
            x, y, w, h = box[index]
            cut = image_pre[y:y+h,x:x+w]
        except:
            return None
        return cut

    def face_box(self,inputImg,scaleF=1.05,minN=5,minSize=(30,30)):
        face_react = []
        if inputImg is not None:
            face_react = self.haar_face.detectMultiScale(inputImg, scaleFactor=scaleF,
                                                         minNeighbors=minN, minSize=minSize,
                                                         flags=cv2.CASCADE_SCALE_IMAGE)
        return face_react

    def preprocess(self, image):

        if image is None:
            return None


        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        face_B = self.face_box(gray)

        if len(face_B) > 0:
            #image = self.cut_face(image, face_B, index=0)
            image = cv2.resize(image, (128, 128))
            image = image.transpose(2, 0, 1)
            image = np.expand_dims(image, 0)
            return np.array(image).astype(np.float32)
        else:
            return None


    def inference(self, image):

        img = self.preprocess(image)

        if img is None:
            return None

        pred = self.sess.run(None, {self.inputs[0]: img})
        pred = np.array(pred).squeeze()
        res = np.argmax(pred, axis=0)

        if res == 1:
            return False
        else:
            return True






