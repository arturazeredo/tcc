from src.utils import image_resize
import mediapipe as mp
import math
import cv2

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)


class GooMedia:
    def __init__(self,size=480, max_num_faces=1, min_detection=0.5):
        self.size = size
        self.static_image_mode = True
        self.max_num_faces = max_num_faces
        self.refine_landmarks = True
        self.min_detection = min_detection

    def preprocess(self,inputImg):
        img_r = image_resize(inputImg,max_=self.size)
        img_p = cv2.cvtColor(img_r, cv2.COLOR_BGR2RGB)
        return img_r, img_p

    def is_valid(self,value):
        return (value > 0 or math.isclose(0, value)) and (value < 1 or
                                                        math.isclose(1, value))

    def points_normalize(self, landmask_list, shape, index=0):
        points = []

        if len(landmask_list) < index+1:
            return points

        h, w = shape[:2]

        for landmark in landmask_list[index].landmark:

            n_x = landmark.x
            n_y = landmark.y

            if not (self.is_valid(n_x) and self.is_valid(n_y)):
                return []

            x_px = min(math.floor(n_x * w), w - 1)
            y_px = min(math.floor(n_y * h), h - 1)
            points.append((x_px,y_px))

        return points

    def extract(self,inputImg):

        img_r, img_p = self.preprocess(inputImg)

        with mp_face_mesh.FaceMesh(
                static_image_mode=self.static_image_mode,
                max_num_faces=self.max_num_faces,
                refine_landmarks=self.refine_landmarks,
                min_detection_confidence=self.min_detection) as face_mesh:

            results = face_mesh.process(img_p)

            if not results.multi_face_landmarks:
                return img_r, []
            else:
                return img_r, results.multi_face_landmarks

    def draw_face(self,image,points):

        if points is not None:
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=points,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())

        return image
