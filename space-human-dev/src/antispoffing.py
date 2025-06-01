import cv2
import numpy as np
import onnxruntime as onnx
from src.utils import softmax

class Anti:
    def __init__(self):
        self.sess1 = onnx.InferenceSession("src/models/anti2")
        self.sess2 = onnx.InferenceSession("src/models/anti3")
        self.inputs1 = [n.name for n in self.sess1.get_inputs()]
        self.inputs2 = [n.name for n in self.sess2.get_inputs()]
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
            image = self.cut_face(image, face_B, index=0)
            image = cv2.resize(image, (80, 80))
            image = image.transpose(2, 0, 1)
            image = np.expand_dims(image, 0)
            return np.array(image).astype(np.float32)
        else:
            return None


    def inference(self, image):

        img = self.preprocess(image)

        if img is None:
            return None

        pred1 = self.sess1.run(None, {self.inputs1[0]: img})
        pred2 = self.sess2.run(None, {self.inputs2[0]: img})
        prediction = np.zeros((1, 3))
        pred1 = softmax(pred1[0])
        pred2 = softmax(pred2[0])
        prediction += pred2 + pred1

        label = np.argmax(prediction)
        value = prediction[0][label] / 2

        if label == 1:
            return 1
        else:
            return 0






