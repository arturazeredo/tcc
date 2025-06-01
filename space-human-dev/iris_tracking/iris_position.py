 # Importing libraries
import cv2 as cv
import numpy as np 
import mediapipe as mp
import math

mp_face_mesh = mp.solutions.face_mesh
# List of eyes and iris
RIGHT_IRIS = [474, 475, 476, 477]
LEFT_IRIS = [469, 470, 471, 472]
L_H_LEFT = [33]  # right eye right most landmark
L_H_RIGHT = [133]  # right eye left most landmark
R_H_LEFT = [362]  # left eye right most landmark
R_H_RIGHT = [263]  # left eye left most landmark
####################################################################################################################

def euclidean_distance(point1, point2): 
    x1, y1 = point1.ravel()
    x2, y2 = point2.ravel()
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distance
def iris_position(iris_center, right_point, left_point):
    center_to_right_dist = euclidean_distance(iris_center, right_point)
    center_to_left_dist = euclidean_distance(iris_center, left_point)
    total_distance = euclidean_distance(right_point, left_point)
    ratio = center_to_right_dist/total_distance
    iris_position = []
    if ratio <= 0.42:
        iris_position = "Right"
    elif ratio > 0.42 and ratio <= 0.57:
        iris_position = "Center"
    else:
        iris_position = "Left"
    return iris_position, ratio



cap = cv.VideoCapture(1) # Open default camera
with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True, # 478 landmarks instead of 468
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:
    while True:
        ret, frame = cap.read()
        if not ret: # check if frame is not read correctly
            break
        frame = cv.flip(frame, 1) # mirror the image
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        img_h, img_w = frame.shape[:2]
        results = face_mesh.process(rgb_frame)
        if results.multi_face_landmarks:
            #print(results.multi_face_landmarks[0].landmark) # 0 index means the eyes landmark (?)
            mesh_points = np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int) for p in results.multi_face_landmarks[0].landmark]) # numpy array for the points
            #print(mesh_points.shape())
            #cv.polylines(frame, [mesh_points[LEFT_EYE]], True, (0, 255, 0), 1, cv.LINE_AA)
            #cv.polylines(frame, [mesh_points[RIGHT_EYE]], True, (0, 255, 0), 1, cv.LINE_AA)
            (l_cx, l_cy), l_radius = cv.minEnclosingCircle(mesh_points[LEFT_IRIS]) # center point left iris
            (r_cx, r_cy), r_radius = cv.minEnclosingCircle(mesh_points[RIGHT_IRIS]) # center point right iris
            center_left = np.array([l_cx, l_cy], dtype=np.int32)
            center_right = np.array([r_cx, r_cy], dtype=np.int32)
            cv.circle(frame, center_left, int(l_radius), (255, 0, 255), 1, cv.LINE_AA)
            cv.circle(frame, center_right, int(r_radius), (255, 0, 255), 1, cv.LINE_AA)
            cv.circle(frame, mesh_points[R_H_RIGHT[0]], 2, (255, 255, 255), -1, cv.LINE_AA)
            cv.circle(frame, mesh_points[R_H_LEFT[0]], 2, (0, 255, 255), -1, cv.LINE_AA)
            cv.circle(frame, mesh_points[L_H_RIGHT[0]], 2, (255, 255, 255), -1, cv.LINE_AA)
            cv.circle(frame, mesh_points[L_H_LEFT[0]], 2, (0, 255, 255), -1, cv.LINE_AA)

            iris_pos_r, ratio = iris_position(center_right, mesh_points[R_H_RIGHT[0]], mesh_points[R_H_LEFT[0]])
            iris_pos_l, ratio = iris_position(center_left, mesh_points[L_H_RIGHT[0]], mesh_points[L_H_LEFT[0]])
            cv.putText(frame, 
                       f"Iris pos Right eye: {iris_pos_r} {ratio:.2f}", 
                       (30, 30), 
                       cv.FONT_HERSHEY_PLAIN, 
                       1.2, 
                       (0, 255, 0),
                       1, 
                       cv.LINE_AA
                       )
            cv.putText(frame, 
                       f"Iris pos Left eye: {iris_pos_l} {ratio:.2f}", 
                       (30, 60), 
                       cv.FONT_HERSHEY_PLAIN, 
                       1.2, 
                       (0, 255, 0),
                       1, 
                       cv.LINE_AA
                       )
        cv.imshow('img', frame)
        key = cv.waitKey(1)
        if key == ord('q') or key == ord('Q'): # wait until 'q' key is pressed
            break
cap.release()
cv.destroyAllWindows()
