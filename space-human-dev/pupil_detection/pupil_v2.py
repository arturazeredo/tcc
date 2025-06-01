import cv2
import mediapipe as mp
import numpy as np

def calculate_EAR(eye_points):
    eye_points = np.array(eye_points)
    A = np.linalg.norm(eye_points[1] - eye_points[5])
    B = np.linalg.norm(eye_points[2] - eye_points[4])
    C = np.linalg.norm(eye_points[0] - eye_points[3])
    ear = (A + B) / (2.0 * C)
    return ear

def calculate_iris_size(landmarks, image_shape, iris_indices):
    image_height, image_width = image_shape[:2]
    iris_landmarks = [landmarks.landmark[i] for i in iris_indices]
    iris_points = [(p.x * image_width, p.y * image_height) for p in iris_landmarks]
    left_point = np.array(iris_points[0])
    right_point = np.array(iris_points[2])
    iris_diameter = np.linalg.norm(right_point - left_point)
    return iris_diameter

def detect_pupil(eye_gray, eye_color, iris_diameter):
    # Use dynamic thresholding
    avg_intensity = np.mean(eye_gray)
    threshold_value = avg_intensity * 0.8  # Adjust multiplier as needed
    _, thresh = cv2.threshold(eye_gray, threshold_value, 255, cv2.THRESH_BINARY_INV)

    # Remove small noises and fill holes
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.medianBlur(thresh, 5)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Find the largest contour
        c = max(contours, key=cv2.contourArea)
        # Calculate the moments to estimate the center of the contour
        M = cv2.moments(c)
        if M['m00'] != 0:
            cX = int(M['m10'] / M['m00'])
            cY = int(M['m01'] / M['m00'])
            center = (cX, cY)
            # Approximate the radius based on contour area
            area = cv2.contourArea(c)
            radius = int(np.sqrt(area / np.pi))
            # Set reasonable constraints for the pupil size
            if radius > 1 and radius < iris_diameter / 2:
                normalized_pupil_size = (2 * radius) / iris_diameter
                cv2.circle(eye_color, center, radius, (0, 255, 0), 2)
                print(f"Normalized Pupil Size: {normalized_pupil_size:.2f}")
            else:
                print("Pupil size out of expected range.")
        else:
            print("Invalid contour moments.")
    else:
        print("No pupil detected.")

    # For debugging: display the thresholded image and contours
    cv2.imshow('Thresholded Eye', thresh)
    cv2.imshow('Cropped Eye Region', eye_color)
    cv2.drawContours(eye_color, contours, -1, (255, 0, 0), 1)
    print(f"Number of contours detected: {len(contours)}")
    for idx, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        print(f"Contour {idx}: Area = {area}")

    return eye_color

def get_eye_region(image, face_landmarks, eye_indices):
    image_height, image_width = image.shape[:2]
    # Get the coordinates of the eye landmarks
    eye_landmarks = [face_landmarks.landmark[i] for i in eye_indices]
    eye_points = np.array([
        (int(p.x * image_width), int(p.y * image_height)) for p in eye_landmarks
    ], dtype=np.int32)

    # Calculate the bounding rectangle of the eye
    x, y, w, h = cv2.boundingRect(eye_points)

    # Crop the eye region from the original image
    eye_region = image[y:y+h, x:x+w]

    # Adjust the eye landmarks to the cropped region
    eye_points_cropped = eye_points - [x, y]
    print("Original Eye Landmarks:", eye_points)
    print("Adjusted Eye Landmarks:", eye_points_cropped)

    return eye_region, eye_points_cropped
def main():
    mp_face_mesh = mp.solutions.face_mesh

    cap = cv2.VideoCapture(0)

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:

        EAR_THRESHOLD = 0.25  # Adjusted threshold

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            image_height, image_width, _ = frame.shape

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Left eye indices for EAR and eye region
                    left_eye_indices = [33, 160, 158, 133, 153, 144]
                    left_eye_points = [
                        (face_landmarks.landmark[i].x * image_width, face_landmarks.landmark[i].y * image_height)
                        for i in left_eye_indices
                    ]
                    ear = calculate_EAR(left_eye_points)

                    if ear > EAR_THRESHOLD:
                        # Eye is open, proceed with pupil detection
                        # Iris landmarks
                        left_iris_indices = [469, 470, 471, 472, 468]
                        iris_diameter = calculate_iris_size(face_landmarks, frame.shape, left_iris_indices)

                        # Get eye region and adjusted landmarks
                        left_eye_region, eye_points_cropped = get_eye_region(frame, face_landmarks, left_eye_indices)

                        if left_eye_region.size != 0:
                            eye_gray = cv2.cvtColor(left_eye_region, cv2.COLOR_BGR2GRAY)
                            eye_color = left_eye_region.copy()

                            # Detect and measure the pupil in the eye ROI
                            pupil_frame = detect_pupil(eye_gray, eye_color, iris_diameter)

                            # For debugging: draw eye landmarks
                            for point in eye_points_cropped:
                                cv2.circle(pupil_frame, tuple(point), 1, (0, 0, 255), -1)

                            # Display the eye with detected pupil
                            cv2.imshow('Left Eye', pupil_frame)
                    else:
                        print("Left eye is closed, skipping pupil detection.")

                    # Break after processing one face
                    break

            cv2.imshow('Frame', frame)
            

            if cv2.waitKey(1) & 0xFF == ord('q') or cv2.waitKey(1) & 0xFF == ord('Q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()