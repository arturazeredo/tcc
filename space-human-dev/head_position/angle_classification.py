import cv2
import numpy as np
from src.mediapipe import GooMedia
from src.visualize import draw_face_one_color

def angle(left_hemisphere_ratio, right_hemisphere_ratio):
    under_limit_centralized = 0.8
    upper_limit_centralized = 1.2
    upper_limit_angled = 1.5

    if under_limit_centralized <= left_hemisphere_ratio <= upper_limit_centralized and under_limit_centralized <= right_hemisphere_ratio <= upper_limit_centralized:
        return 'The face is centralized'
    elif upper_limit_centralized < left_hemisphere_ratio < upper_limit_angled:
        return 'The face is a little bit more on the left hemisphere.'
    elif upper_limit_centralized < right_hemisphere_ratio < upper_limit_angled:
        return 'The face is a little bit more on the right hemisphere.'
    elif left_hemisphere_ratio >= upper_limit_angled:
        return 'The face is more angled on the left hemisphere.'
    elif right_hemisphere_ratio >= upper_limit_angled:
        return 'The face is more angled on the right hemisphere.'

def main():
    pface = GooMedia(1080)  # Extrator dos pontos
    cap = cv2.VideoCapture(0)  # Open the default camera

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Resize frame
        #frame_r = cv2.resize(frame, (480, 640))
        frame_r = frame
        # Extract landmarks
        frame_resize, landmask = pface.extract(frame_r)

        if landmask:
            points = pface.points_normalize(landmask, frame_r.shape)

            # Find bounding box
            max_y = max(point[1] for point in points)
            min_y = min(point[1] for point in points)
            max_x = max(point[0] for point in points)
            min_x = min(point[0] for point in points)

            # Calculate midpoint
            mid_x = (max_x + min_x) // 2

            # Count points in each hemisphere
            left_hemisphere_count = sum(1 for point in points if point[0] < mid_x)
            right_hemisphere_count = len(points) - left_hemisphere_count

            # Calculate area and densities
            area = (max_x - min_x) * (max_y - min_y)
            points_density = len(points) / area
            left_hemisphere_density = left_hemisphere_count / (area/2)
            right_hemisphere_density = right_hemisphere_count / (area/2)

            # Calculate ratios
            left_hemisphere_ratio = left_hemisphere_density / points_density
            right_hemisphere_ratio = right_hemisphere_density / points_density

            # Get angle message
            check_angle = angle(left_hemisphere_ratio, right_hemisphere_ratio)

            # Draw face points
            points_face = draw_face_one_color(frame_resize, points)

            # Draw bounding box and midline
            cv2.rectangle(points_face, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)
            cv2.line(points_face, (mid_x, min_y), (mid_x, max_y), (0, 0, 255), 2)

            # Add message to image
            cv2.putText(points_face, check_angle, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Show landmask
            landmask_image = np.zeros_like(points_face)
            for point in points:
                x_px, y_px = point
                landmask_image[y_px, x_px] = 255
            landmask_image = 255 - landmask_image

            # Display the results
            cv2.imshow('Face Analysis', points_face)
            cv2.imshow('Landmask', landmask_image)

        # Check for 'q' key press to quit
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("'q' key pressed. Exiting...")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()