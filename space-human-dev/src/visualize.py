import cv2
import random

import numpy as np

class_color = {0: (76, 69, 157), 1: (113, 120, 132), 2: (201, 246, 194), 3: (62, 196, 113),
               4: (39, 109, 72),  5: (201, 230, 234),6: (62, 136, 111), 7: (7, 209, 14),
               8: (145, 212, 58), 9: (168, 247, 21), 10: (67, 98, 152), 11: (199, 42, 167),
               12: (153, 0, 234), 13: (65, 177, 74), 14: (114, 128, 174), 15: (29, 87, 246),
               16: (35, 158, 254), 17: (200, 213, 161), 18: (59, 167, 147), 19: (47, 159, 48),
               20: (47, 76, 110), 21: (2, 67, 214), 22: (18, 10, 221), 23: (248, 36, 239),
               24: (236, 222, 133), 25: (191, 247, 120), 26: (208, 3, 231), 27: (67, 147, 101),
               28: (142, 205, 190), 29: (79, 80, 241), 30: (84, 76, 234), 31: (27, 132, 83),
               32: (134, 240, 140), 33: (182, 172, 24), 34: (124, 200, 56), 35: (218, 57, 156),
               36: (157, 203, 85), 37: (190, 214, 87), 38: (75, 40, 66), 39: (194, 14, 84),
               40: (251, 154, 108), 41: (78, 225, 5), 42: (98, 82, 83), 43: (175, 194, 175),
               44: (128, 2, 230), 45: (255, 37, 119), 46: (250, 79, 167), 47: (133, 241, 223),
               48: (38, 168, 143), 49: (27, 206, 240), 50: (127, 173, 98), 51: (219, 214, 92),
               52: (168, 129, 195), 53: (67, 139, 14), 54: (18, 90, 135), 55: (38, 34, 178),
               56: (100, 200, 152), 57: (109, 128, 184), 58: (157, 232, 37), 59: (44, 230, 244),
               60: (210, 126, 130), 61: (124, 190, 163), 62: (176, 198, 189), 63: (24, 105, 252),
               64: (1, 137, 33), 65: (236, 81, 75), 66: (225, 135, 127), 67: (70, 216, 159)}


def draw_face(frame, points, box,show_number=False,size_front=0.2):

    if len(points) > 0:
        (x,y,w, h) = box
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 1)

        for i, (x,y) in enumerate(points):
            if show_number:
                frame = cv2.putText(frame, str(i+1), (x,y), cv2.FONT_HERSHEY_SIMPLEX,
                                    size_front, class_color[i], 1, cv2.LINE_AA)
            cv2.circle(frame, (x, y), 1, class_color[i], -1)

    return frame

def draw_face_one_color(frame, points,show_number=False,size_front=0.2,color=(0,255,0)):

    if len(points) > 0:

        for i, (x,y) in enumerate(points):
            if show_number:
                frame = cv2.putText(frame, str(i+1), (x,y), cv2.FONT_HERSHEY_SIMPLEX,
                                    size_front, color, 1, cv2.LINE_AA)
            cv2.circle(frame, (x, y), 1, color, -1)

    return frame

def draw_face_one_points(frame, points,color=(0,255,0),show_number=True,size_front=0.2,rad=1):

    if len(points) > 0:
        for i, (x,y) in enumerate(points):
            if show_number:
                frame = cv2.putText(frame, str(i + 1), (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                                    size_front, (255,0,0), 1, cv2.LINE_AA)
            cv2.circle(frame, (x, y), rad, color, -1)

    return frame


def generation_color_vector(max=68):
    dict = {}

    for i in range(max):

        dict[i] = (random.randint(0,255),
                   random.randint(0,255),
                   random.randint(0,255))
    return dict



def draw_face_to_box(img_r,face,eyer,eyel):
    frame = img_r.copy()

    print("FACE: {}, EYES_R: {}, EYES_L: {}".format(len(face), len(eyer), len(eyel)))

    if len(face) > 0:
        (fx, fy, fw, fh) = face[0]
        cv2.rectangle(frame,(fx,fy), (fx+fw,fy+fh), (0,255,0),1)

        if len(eyer) > 0:
            (x, y, w, h) = eyer[0]
            cv2.rectangle(frame, (fx + x, fy + y), (fx + x + w, fy + y + h), (255,125,0),1)

        if len(eyel) > 0:
            (x, y, w, h) = eyel[0]
            cv2.rectangle(frame, (fx + x, fy + y), (fx + x + w, fy + y + h), (125,125,0),1)


    return frame


def draw_shape(img_r, shape,cor=(0,255,0)):
    if len(shape) > 0:
        area = cv2.contourArea(shape)

        img_r = cv2.putText(img_r, "Area: {}".format(area), (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.drawContours(img_r, [shape], 0, cor, -1)
        x, y, w, h = cv2.boundingRect(shape)
        cv2.rectangle(img_r, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return img_r


def frame_oculto(img_r):
    info = img_r.shape
    new_img = np.ones(info,dtype="uint8")
    return new_img

def draw_text(image,text,diff=(50,50),font_size=0.5, cor=(0,0,255)):

    #print(text)
    image = cv2.putText(image, text, diff, cv2.FONT_HERSHEY_SIMPLEX,
                        font_size, cor, 1, cv2.LINE_AA)

    return image

def draw_circule(image, central, color=(255,0,0)):
    h, w = image.shape[:2]

    min_x = w*0.1
    min_y = h*0.1
    max_x = int(w-(w*0.1))
    max_y = int(h-(h*0.1))
    print(central)
    cx, cy = central

    if cx < min_x:
        cx = min_x

    if cy < min_y:
        cy = min_y

    if cx > max_x:
        cx = max_x

    if cy > max_y:
        cy = max_y

    raio = int(w/8)

    new_frame = cv2.circle(image, (cx, cy), raio, color, -1)
    return new_frame
