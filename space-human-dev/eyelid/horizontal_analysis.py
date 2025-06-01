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


def draw_circle(frame, x_position):
    height, width = frame.shape[:2]
    mirrored_x = 1.0 - x_position
    center = (int(width * mirrored_x), height // 2)
    cv2.circle(frame, center, 30, (0, 255, 0), -1)
    return frame, center

def compare_gaze(frame1, frame2, p1, p2):
    pface = GooMedia(1080)

    _, landmask1 = pface.extract(frame1)
    points1 = pface.points_normalize(landmask1, frame1.shape)
    eye1 = get_points_media(points1, 'iris_analyser2')

    _, landmask2 = pface.extract(frame2)
    points2 = pface.points_normalize(landmask2, frame2.shape)
    eye2 = get_points_media(points2, 'iris_analyser2')

    distance1 = distance_entre_points(eye1, p1, p2)
    distance2 = distance_entre_points(eye2, p1, p2)

    if distance1 is None or distance2 is None:
        print("Error: Unable to compare gaze due to missing data.")
        return None

    return distance1 > distance2

def display_debug_frames(captured_frames):
    if not captured_frames:
        return
    
    # Calculate the dimensions for the debug window
    frame_height, frame_width = captured_frames[0].shape[:2]
    total_width = frame_width * len(captured_frames)
    
    # Create a canvas to hold all frames side by side
    debug_canvas = np.zeros((frame_height, total_width, 3), dtype=np.uint8)
    
    # Add text labels for each frame
    labels = ['Left', 'Center', 'Right']
    
    # Place each frame and its label on the canvas
    for i, frame in enumerate(captured_frames):
        x_offset = i * frame_width
        debug_canvas[:, x_offset:x_offset + frame_width] = frame
        
        # Add label
        if i < len(labels):
            cv2.putText(debug_canvas, labels[i], 
                       (x_offset + 10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       1, (0, 255, 0), 2)
    
    # Display the debug window
    cv2.imshow('Captured Frames Debug', debug_canvas)
    cv2.waitKey(1)

def main():
    pface = GooMedia(1080)
    cap = cv2.VideoCapture(0)

    positions = [0.1, 0.5, 0.9]  # Left, Center, Right
    captured_frames = []
    current_position = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        frame = cv2.flip(frame, 1)
        frame_r = frame.copy()
        
        frame_resize, landmask = pface.extract(frame_r)
        points = pface.points_normalize(landmask, frame_r.shape)

        points_face = draw_face_one_color(frame_resize, points)

        frame_with_circle, circle_center = draw_circle(frame_resize, positions[current_position])
        
        # Display the main frame
        cv2.imshow('Eye Challenge', frame_with_circle)
        
        # Display debug frames if any are captured
        if captured_frames:
            display_debug_frames(captured_frames)

        def mouse_callback(event, x, y, flags, param):
            nonlocal current_position, captured_frames
            if event == cv2.EVENT_LBUTTONDOWN:
                if (x - circle_center[0])**2 + (y - circle_center[1])**2 <= 30**2:
                    captured_frames.append(frame_resize.copy())
                    current_position += 1
                    if current_position >= len(positions):
                        compare_frames(captured_frames)

        cv2.setMouseCallback('Eye Challenge', mouse_callback)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("'q' key pressed. Exiting...")
            break

        if current_position >= len(positions):
            break

    cap.release()
    cv2.destroyAllWindows()

def compare_frames(captured_frames):
    result = compare_gaze(captured_frames[0], captured_frames[1], 9, 11)
    if result is not None:
        result1 = result
        result = compare_gaze(captured_frames[1], captured_frames[2], 9, 11)
        if result is not None:
            result2 = result

            result_frame = np.zeros((600, 1000, 3), dtype=np.uint8)

            if result1:
                print("Eyes are more angled at the Right compared to the Center")
            else:
                print("Eyes are more centralized at the Right compared to the Center - UNCOMMON")

            if result2:
                print("Eyes are more angled at the Center compared to the Left - UNCOMMON")
            else:
                print("Eyes are more centralized at the Center compared to the Left")

            cv2.putText(result_frame, "Gaze Comparison:", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(result_frame, "Right vs Center:", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(result_frame, "Eyes are more angled at the Right compared to the Center", (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(result_frame, "Center vs Left:", (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(result_frame, "Eyes are more angled at the Left compared to the Center", (50, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            cv2.putText(result_frame, "Interpretation:", (50, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(result_frame, "This comparison helps understand how eye gaze is angled", (50, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(result_frame, "changes when looking at different horizontal positions", (50, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            cv2.imshow('Gaze Comparison', result_frame)
            cv2.waitKey(0)

        else:
            print("Failed to compare frames Right vs Center")
    else:
        print("Error: Unable to compare frames Left vs Center")

if __name__ == "__main__":
    main()