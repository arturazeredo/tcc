import cv2 as cv
import mediapipe as mp
import numpy as np
import time

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

mp_drawning = mp.solutions.drawing_utils

drawing_spec = mp_drawning.DrawingSpec(thickness=1, circle_radius=1) # drawing specifications

cap = cv.VideoCapture(0) # 0 for webcam, 1 for external camera

while cap.isOpened():
    sucess, image = cap.read()

    start = time.time()

    # Flip the image horizontally for a later selfie-view display
    # Convert color from BGR to RGB
    image = cv.cvtColor(cv.flip(image, 1), cv.COLOR_BGR2RGB)
    # Improve performance
    image.flags.writeable = False

    # Get result
    results = face_mesh.process(image)

    # Draw the face mesh annotations on the image.
    #image.flags.writeable = True
    image = cv.cvtColor(image, cv.COLOR_RGB2BGR)

    # Create a copy of the image
    draw_image = np.copy(image)

    img_h, img_w, img_c = image.shape
    face_3d = []
    face_2d = []

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for idx, lm in enumerate(face_landmarks.landmark): # lm means landmarks
                if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199: # interest points
                    if idx == 1:
                        nose_2d = (lm.x * img_w, lm.y * img_h)
                        nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)

                    x, y = int(lm.x * img_w), int(lm.y * img_h) # face coordinates

                    # Get the 2D Coordinates
                    face_2d.append([x, y])

                    # Get the 3D Coordinates
                    face_3d.append([x, y, lm.z]) # z value

            # Convert it to the NumPy array
            face_2d = np.array(face_2d, dtype=np.float64)

            # Convert it to the NumPy array
            face_3d = np.array(face_3d, dtype=np.float64)

            # Camera matrix
            focal_lenght = 1 * img_w

            # The camera matrix is a 3x3 matrix that maps 3D points
            # in the world coordinates to 2D points in the image
            # coordinates. The matrix is composed of the focal length
            # (fx, fy) and the principal point (cx, cy) of the camera.
            #
            # In this case, we are using a simple camera model with
            # a square pixel aspect ratio, so the focal length is the
            # same in the x and y directions (fx = fy). The principal
            # point is set to the center of the image (cx = img_w / 2,
            # cy = img_h / 2).

            cam_matrix = np.array([
                [focal_lenght, 0, img_h / 2],
                [0, focal_lenght, img_w / 2],
                [0, 0, 1]
            ])

            # Distortion parameters
            dist_matrix = np.zeros((4, 1), dtype=np.float64)

            # Solve Perspective-n-Point problem. The Perspective-n-Point (PnP) problem is a classic problem in Computer Vision, 
            # which is used to determine the 3D pose of an object from 2D image points. 
            # The PnP algorithm is used to find the rotation and translation vectors between the object's local coordinate system and the camera's coordinate system.
            #  In this case, we are using the solvePnP function from OpenCV to solve the PnP problem 
            # and find the rotation and translation vectors between the face's local coordinate system and the camera's coordinate system.
            # Solve PnP 
            sol_flag, rot_vec, trans_vec = cv.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

            # Get rotational matrix
            rmat, jac = cv.Rodrigues(rot_vec)

            # Get angles
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv.RQDecomp3x3(rmat)

            # Get the y rotation degree
            x = angles[0] * 360
            y = angles[1] * 360
            z = angles[2] * 360

            # See where the user's head tilting
            if y < -10:
                text = "Looking: Left"
            elif y > 10:
                text = "Looking: Right"
            elif x < -10:
                text = "Looking: Down"
            elif x > 10:
                text = "Looking: Up"
            else:
                text = "Looking: Forward"

            # Display nose direction
            nose_3d_projection, jacobian = cv.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)

            p1 = (int(nose_2d[0]), int(nose_2d[1]))
            p2 = (int(nose_2d[0] + y * 10), int(nose_2d[1] - x * 10))

            # Add the text on the image
            cv.line(draw_image, p1, p2, (255, 0, 0), 3)

            cv.putText(draw_image, text, (20, 50), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
            cv.putText(draw_image, f"x: {np.round(x,2)}", (500, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv.putText(draw_image, f"y: {np.round(y,2)}", (500, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv.putText(draw_image, f"z: {np.round(z,2)}", (500, 150), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        end = time.time()
        totalTime = end - start

        fps = 1 / totalTime

        cv.putText(draw_image, f'FPS: {int(fps)}', (20, 450), cv.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

        mp_drawning.draw_landmarks(
            image=draw_image,
            landmark_list = face_landmarks,
            connections = mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec = drawing_spec,
            connection_drawing_spec = drawing_spec
        )
    cv.imshow('Head Pose Estimation', draw_image)

    if cv.waitKey(5) & 0xFF == ord('q') or cv.waitKey(5) & 0xFF == ord('Q'):
        break

cap.release() 

