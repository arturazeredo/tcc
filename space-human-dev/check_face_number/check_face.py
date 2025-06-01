from src.mediapipe import GooMedia
import cv2
import numpy as np
from src.visualize import *
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from src.utils import image_resize
from PIL import Image

def check_number_face(landmasks):
    if len(landmasks) == 1:
        return 'Número de faces correto'
    else:
        return 'Apresente somente um rosto na câmera'

def main():
    pface = GooMedia(1080, 2) # Extrator dos pontos
    cap = cv2.VideoCapture(0) # Open the default camera
    
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

        
        points = pface.points_normalize(landmask, frame_r.shape)

        # Draw face points
        points_face = draw_face_one_color(frame_resize, points)

        # Add message to image
        cv2.putText(points_face, check_number_face(landmask), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

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