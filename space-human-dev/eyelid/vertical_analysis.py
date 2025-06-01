import sys
import os
currentdir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(currentdir, '..'))
sys.path.append('/eyelid')
sys.path.insert(0, parent_dir)  # add grandparent directory to sys.path
from src.mediapipe import GooMedia
import cv2
import numpy as np
from src.visualize import *
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from src.utils import image_resize
from src.geometry import get_points_media
from src.geometry import distance_entre_points
from PIL import Image

def media_distance(points):
    eyelid_l = [28, 29, 30, 159, 160, 161]
    eyelid_r = [258, 259, 260, 386, 387, 388]
    
    def calculate_eyelid_distance(coords):
        pairs = [(coords[2], coords[5]), (coords[0], coords[4]), (coords[1], coords[3])]
        distances = []
        for p1, p2 in pairs:
            try:
                distance = distance_entre_points(points, p1, p2)
                distances.append(distance)
            except IndexError:
                print(f"IndexError: Point {p1} or {p2} is out of range.")
                print(f"Total number of points: {len(points)}")
                print(f"Attempting to access indices: {p1} and {p2}")
                return None
        return np.mean(distances) if distances else None
    
    left_distance = calculate_eyelid_distance(eyelid_l)
    right_distance = calculate_eyelid_distance(eyelid_r)
    
    if left_distance is None or right_distance is None:
        print("Error: Unable to calculate distances for one or both eyes.")
        return None
    
    return np.mean([left_distance, right_distance])

def compare_eyelid(frame1, frame2):
    pface = GooMedia(1080)  # Extrator dos pontos

    # Extract landmarks for both frames
    _, landmask1 = pface.extract(frame1)
    points1 = pface.points_normalize(landmask1, frame1.shape)

    _, landmask2 = pface.extract(frame2)
    points2 = pface.points_normalize(landmask2, frame2.shape)

    # Calculate media distance (average eyelid distance) for both frames
    distance1 = media_distance(points1)
    distance2 = media_distance(points2)

    if distance1 is None or distance2 is None:
        print("Error: Unable to compare eyelids due to missing data.")
        return None

    # Compare the distances
    return distance1 > distance2  # True if eyelid distance is greater in frame1 (eyes more open)

def draw_circle(frame, y_position):
    height, width = frame.shape[:2]
    center = (width // 2, int(height * y_position))
    cv2.circle(frame, center, 30, (0, 255, 0), -1)
    return frame, center

def main():
    pface = GooMedia(1080)  # Extractor dos pontos
    cap = cv2.VideoCapture(0)  # Open the default camera
    
    positions = [0.1, 0.5, 0.9]  # Top, center, bottom
    captured_frames = []
    current_position = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        frame_r = frame.copy()
        
        # Extract landmarks
        frame_resize, landmask = pface.extract(frame_r)
        points = pface.points_normalize(landmask, frame_r.shape)

        # Print the number of detected points for debugging
        print(f"Number of points detected: {len(points)}")

        # Draw face points
        points_face = draw_face_one_color(frame_resize, points)
        
        # Draw circle at current position
        frame_with_circle, circle_center = draw_circle(frame_resize, positions[current_position])
        
        # Display the frame
        cv2.imshow('Eye Challenge', frame_with_circle)

        # Check for mouse click
        def mouse_callback(event, x, y, flags, param):
            nonlocal current_position, captured_frames
            if event == cv2.EVENT_LBUTTONDOWN:
                if (x - circle_center[0])**2 + (y - circle_center[1])**2 <= 30**2:
                    captured_frames.append(frame_resize)
                    current_position += 1
                    if current_position >= len(positions):
                        compare_frames(captured_frames)

        cv2.setMouseCallback('Eye Challenge', mouse_callback)

        # Check for 'q' key press to quit
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("'q' key pressed. Exiting...")
            break

        if current_position >= len(positions):
            break

    cap.release()
    cv2.destroyAllWindows()

def compare_frames(frames):
    result = compare_eyelid(frames[0], frames[1])
    if result is not None:
        result1 = result
        result = compare_eyelid(frames[1], frames[2])
        if result is not None:
            result2 = result
            
            result_frame = np.zeros((600, 1000, 3), dtype=np.uint8)
            
            # Comparison between Top and Center positions
            if result1:
                top_center_text = "Eyes are more open at the Top compared to the Center"
            else:
                top_center_text = "Eyes are more closed at the Top compared to the Center"
            
            # Comparison between Center and Bottom positions
            if result2:
                center_bottom_text = "Eyes are more open at the Center compared to the Bottom"
            else:
                center_bottom_text = "Eyes are more closed at the Center compared to the Bottom"
            
            # Add explanatory text
            cv2.putText(result_frame, "Eye Openness Comparison:", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(result_frame, "Top vs Center:", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(result_frame, top_center_text, (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(result_frame, "Center vs Bottom:", (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(result_frame, center_bottom_text, (50, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            cv2.putText(result_frame, "Interpretation:", (50, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(result_frame, "- 'More open' means the eyelids are further apart", (50, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(result_frame, "- 'More closed' means the eyelids are closer together", (50, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(result_frame, "This comparison helps understand how eye openness", (50, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(result_frame, "changes when looking at different vertical positions", (50, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            cv2.imshow('Results', result_frame)
            cv2.waitKey(0)
        else:
            print("Error: Unable to compare Center vs Bottom frames.")
    else:
        print("Error: Unable to compare Top vs Center frames.")

if __name__ == "__main__":
    main()