import cv2
import mediapipe as mp
import numpy as np

def calculate_eye_size(landmarks, image_shape, eye_indices):
    image_height, image_width = image_shape[:2]
    eye_landmarks = [landmarks.landmark[i] for i in eye_indices]
    eye_points = [(p.x * image_width, p.y * image_height) for p in eye_landmarks]
    # Calculate eye width
    left_point = np.array(eye_points[0])
    right_point = np.array(eye_points[3])
    eye_width = np.linalg.norm(right_point - left_point)
    return eye_width

def dynamic_threshold(eye_gray):
    # Compute the average intensity
    avg_intensity = np.mean(eye_gray)
    # Set threshold value based on average intensity
    threshold_value = avg_intensity * 0.8
    # Apply thresholding
    _, thresh = cv2.threshold(eye_gray, threshold_value, 255, cv2.THRESH_BINARY_INV)
    return thresh

def detect_pupil(eye_gray, eye_color, eye_width, image):
    # Apply a Gaussian blur to reduce noise
    blur = cv2.GaussianBlur(eye_gray, (7, 7), 0)

    # Apply thresholding to create a binary image
    thresh = dynamic_threshold(eye_gray)

    # Remove small noises and fill holes using morphological operations
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.erode(thresh, kernel, iterations=2)
    thresh = cv2.dilate(thresh, kernel, iterations=4)
    thresh = cv2.medianBlur(thresh, 5)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # If contours are found, process them
    if contours:
        # Find the largest contour (assuming it's the pupil)
        c = max(contours, key=cv2.contourArea)

        # Calculate the center and radius of the minimum enclosing circle around the contour
        (x, y), radius = cv2.minEnclosingCircle(c)
        center = (int(x), int(y))
        radius = int(radius)

        # Calculate normalized pupil size
        normalized_pupil_size = radius / (eye_width / 2)

        # Draw the circle on the eye_color image
        cv2.circle(eye_color, center, radius, (0, 255, 0), 2)
        # Print the normalized pupil size
        cv2.putText(image, f"Normalized Pupil Size: {normalized_pupil_size:.2f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    else:
        cv2.putText(image, "No pupil detected.", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    return eye_color

def get_eye_region(image, face_landmarks, eye_indices):
    image_height, image_width = image.shape[:2]

    # Get the coordinates of the eye landmarks
    eye_landmarks = [face_landmarks.landmark[i] for i in eye_indices]
    eye_points = [(int(p.x * image_width), int(p.y * image_height)) for p in eye_landmarks]

    # Create a mask for the eye region
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.array(eye_points, dtype=np.int32), 255)

    # Extract the eye region from the image
    eye_region = cv2.bitwise_and(image, image, mask=mask)

    # Crop the eye region to the bounding rectangle of the mask
    x_coords = [point[0] for point in eye_points]
    y_coords = [point[1] for point in eye_points]
    x1, x2 = min(x_coords), max(x_coords)
    y1, y2 = min(y_coords), max(y_coords)
    eye_region_cropped = eye_region[y1:y2, x1:x2]

    return eye_region_cropped

def main():
    mp_face_mesh = mp.solutions.face_mesh

    # Start video capture
    cap = cv2.VideoCapture(0)

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,  # Ensures iris landmarks are included
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Flip the frame horizontally for a later selfie-view display
            frame = cv2.flip(frame, 1)
            image_height, image_width, _ = frame.shape

            # Convert the BGR image to RGB before processing
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process the image and find face landmarks
            results = face_mesh.process(rgb_frame)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Get left eye landmarks
                    left_eye_indices = [33, 160, 158, 133]  # Approximate indices for left eye corners
                    left_eye_width = calculate_eye_size(face_landmarks, frame.shape, left_eye_indices)

                    # Get bounding box for left eye
                    left_eye_region = get_eye_region(frame, face_landmarks, left_eye_indices)

                    if left_eye_region.size != 0:
                        eye_gray = cv2.cvtColor(left_eye_region, cv2.COLOR_BGR2GRAY)
                        eye_color = left_eye_region.copy()

                        # Detect and measure the pupil in the eye ROI
                        pupil_frame = detect_pupil(eye_gray, eye_color, left_eye_width, frame)

                        # Display the eye with detected pupil
                        cv2.imshow('Left Eye', pupil_frame)

                    # You can repeat the same process for the right eye if desired

                    # Break after processing one face
                    break

            # Display the frame with annotations
            cv2.imshow('Frame', frame)

            # Exit on pressing 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Release the capture and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()