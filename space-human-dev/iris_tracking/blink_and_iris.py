import cv2 as cv
import numpy as np
import mediapipe as mp
import time
import math

# Constants
FONTS = cv.FONT_HERSHEY_COMPLEX
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]
L_H_LEFT = [33]
L_H_RIGHT = [133]
R_H_LEFT = [362]
R_H_RIGHT = [263]

# Variables
CEF_COUNTER = 0
TOTAL_BLINKS = 0
CLOSED_EYES_FRAME = 3

def landmarksDetection(img, results, draw=False):
    img_height, img_width = img.shape[:2]
    mesh_coord = [(int(point.x * img_width), int(point.y * img_height)) for point in results.multi_face_landmarks[0].landmark]
    if draw:
        [cv.circle(img, p, 2, (0,255,0), -1) for p in mesh_coord]
    return mesh_coord

def euclidean_distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def get_aspect_ratio(landmarks, eye):
    horizontal_right = landmarks[eye[0]]
    horizontal_left = landmarks[eye[8]]
    vertical_top = landmarks[eye[12]]
    vertical_bottom = landmarks[eye[4]]
    
    horizontal_distance = euclidean_distance(horizontal_right, horizontal_left)
    vertical_distance = euclidean_distance(vertical_top, vertical_bottom)
    
    aspect_ratio = horizontal_distance / vertical_distance
    return aspect_ratio

def iris_position(iris_center, right_point, left_point):
    center_to_right_dist = euclidean_distance(iris_center, right_point)
    total_distance = euclidean_distance(right_point, left_point)
    ratio = center_to_right_dist / total_distance
    
    if ratio <= 0.42:
        return "Right", ratio
    elif ratio > 0.42 and ratio <= 0.57:
        return "Center", ratio
    else:
        return "Left", ratio

# Webcam setup
cap = cv.VideoCapture(0)

# Mediapipe FaceMesh
with mp.solutions.face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:

    start_time = time.time()
    frame_counter = 0

    while True:
        frame_counter += 1
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        img_h, img_w = frame.shape[:2]
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            mesh_points = landmarksDetection(frame, results)

            # Blink detection
            left_eye_ratio = get_aspect_ratio(mesh_points, LEFT_EYE)
            right_eye_ratio = get_aspect_ratio(mesh_points, RIGHT_EYE)
            blink_ratio = (left_eye_ratio + right_eye_ratio) / 2

            if blink_ratio > 5.0:
                CEF_COUNTER += 1
                cv.putText(frame, 'Blink', (200, 30), FONTS, 1.3, (255, 0, 255), 2)
            else:
                if CEF_COUNTER > CLOSED_EYES_FRAME:
                    TOTAL_BLINKS += 1
                    CEF_COUNTER = 0

            cv.putText(frame, f'Total Blinks: {TOTAL_BLINKS}', (100, 150), FONTS, 0.6, (0, 0, 255), 2)

            # Iris position detection
            mesh_points = np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int) for p in results.multi_face_landmarks[0].landmark]) # numpy array for the points
            left_iris = mesh_points[LEFT_IRIS]
            right_iris = mesh_points[RIGHT_IRIS]

            # Left eye
            l_cx, l_cy = left_iris.mean(axis=0).astype(int)
            l_left = mesh_points[L_H_LEFT][0]
            l_right = mesh_points[L_H_RIGHT][0]

            # Right eye
            r_cx, r_cy = right_iris.mean(axis=0).astype(int)
            r_left = mesh_points[R_H_LEFT][0]
            r_right = mesh_points[R_H_RIGHT][0]

            # Draw iris centers
            cv.circle(frame, (l_cx, l_cy), 3, (255, 0, 255), -1)
            cv.circle(frame, (r_cx, r_cy), 3, (255, 0, 255), -1)

            # Determine iris positions
            iris_pos_l = iris_position((l_cx, l_cy), l_right, l_left)
            iris_pos_r = iris_position((r_cx, r_cy), r_right, r_left)
            print(iris_pos_l)
            print(iris_pos_r)
            cv.putText(frame, f"Left eye: {iris_pos_l}", (30, 30), cv.FONT_HERSHEY_PLAIN, 1.2, (0, 255, 0), 1, cv.LINE_AA)
            cv.putText(frame, f"Right eye: {iris_pos_r}", (30, 60), cv.FONT_HERSHEY_PLAIN, 1.2, (0, 255, 0), 1, cv.LINE_AA)

        # Calculate and display FPS
        end_time = time.time() - start_time
        fps = frame_counter / end_time
        cv.putText(frame, f'FPS: {round(fps,1)}', (30, 90), FONTS, 0.6, (0, 255, 0), 2)

        cv.imshow('Eye Tracking', frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv.destroyAllWindows()