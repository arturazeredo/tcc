import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Initialize the face mesh model
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def get_face_3d_2d_points(face_landmarks, image_shape):
    """Extract 3D and 2D facial landmarks for pose estimation."""
    img_h, img_w = image_shape[:2]
    face_3d = []
    face_2d = []
    
    for idx, lm in enumerate(face_landmarks.landmark):
        if idx in [1, 33, 61, 199, 263, 291]:
            x, y = int(lm.x * img_w), int(lm.y * img_h)
            face_2d.append([x, y])
            face_3d.append([x, y, lm.z])
    
    return np.array(face_3d, dtype=np.float64), np.array(face_2d, dtype=np.float64)

def estimate_head_pose(face_3d, face_2d, image_shape):
    """Estimate head pose using solvePnP."""
    img_h, img_w = image_shape[:2]
    focal_length = 1 * img_w
    
    cam_matrix = np.array([
        [focal_length, 0, img_h / 2],
        [0, focal_length, img_w / 2],
        [0, 0, 1]
    ])
    
    dist_matrix = np.zeros((4, 1), dtype=np.float64)
    
    success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
    
    rmat, _ = cv2.Rodrigues(rot_vec)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
    
    return angles, rot_vec, trans_vec, cam_matrix, dist_matrix

def estimate_gaze_direction(left_eye, right_eye, head_angles):
    """Estimate gaze direction based on eye positions and head pose."""
    eye_center = ((left_eye[0] + right_eye[0]) / 2, (left_eye[1] + right_eye[1]) / 2)
    eye_vector = (right_eye[0] - left_eye[0], right_eye[1] - left_eye[1])
    
    # Adjust gaze vector based on head pose
    x_angle, y_angle, _ = head_angles
    gaze_vector = (
        -eye_vector[1] * np.cos(y_angle) - eye_vector[0] * np.sin(x_angle),
        eye_vector[0] * np.cos(x_angle) - eye_vector[1] * np.sin(y_angle)
    )
    
    return eye_center, gaze_vector

def project_gaze(eye_center, gaze_vector, image_shape, scale=100):
    """Project gaze direction onto the image."""
    h, w = image_shape[:2]
    gaze_point = (
        int(eye_center[0] + gaze_vector[0] * scale),
        int(eye_center[1] + gaze_vector[1] * scale)
    )
    gaze_point = (max(0, min(w, gaze_point[0])), max(0, min(h, gaze_point[1])))
    return gaze_point

cap = cv2.VideoCapture(1)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = face_mesh.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
            )

            face_3d, face_2d = get_face_3d_2d_points(face_landmarks, image.shape)
            head_angles, rot_vec, trans_vec, cam_matrix, dist_matrix = estimate_head_pose(face_3d, face_2d, image.shape)

            # Get iris landmarks
            h, w, _ = image.shape
            left_iris = (int(face_landmarks.landmark[468].x * w), int(face_landmarks.landmark[468].y * h))
            right_iris = (int(face_landmarks.landmark[473].x * w), int(face_landmarks.landmark[473].y * h))

            # Draw circles for iris centers
            cv2.circle(image, left_iris, 2, (0, 0, 255), -1)
            cv2.circle(image, right_iris, 2, (0, 0, 255), -1)

            # Estimate gaze direction
            eye_center, gaze_vector = estimate_gaze_direction(left_iris, right_iris, head_angles)
            
            # Project gaze onto image
            gaze_point = project_gaze(eye_center, gaze_vector, image.shape)

            # Draw gaze projection
            cv2.line(image, (int(eye_center[0]), int(eye_center[1])), gaze_point, (255, 0, 0), 2)
            cv2.circle(image, gaze_point, 5, (255, 0, 0), -1)

            # Display head pose information
            x, y, z = head_angles
            cv2.putText(image, f"Head Pose: x:{int(x)}, y:{int(y)}, z:{int(z)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow('MediaPipe Face Mesh with Gaze and Head Pose Estimation', image)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()