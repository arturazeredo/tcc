import tkinter as tk
import cv2
from PIL import Image, ImageTk
import angle_classification_optimized_H as horizontal
import angle_classification_optimized_V as vertical

import time

class FaceAngleApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)

        self.video_source = 0
        self.vid = cv2.VideoCapture(self.video_source)

        self.canvas = tk.Canvas(window, width=self.vid.get(cv2.CAP_PROP_FRAME_WIDTH), height=self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.canvas.pack()

        self.btn_horizontal = tk.Button(window, text="Horizontal", width=15, command=self.start_horizontal)
        self.btn_horizontal.pack(side=tk.LEFT, padx=5, pady=5)

        self.btn_vertical = tk.Button(window, text="Vertical", width=15, command=self.start_vertical)
        self.btn_vertical.pack(side=tk.LEFT, padx=5, pady=5)

        self.btn_stop = tk.Button(window, text="Stop", width=15, command=self.stop_analysis)
        self.btn_stop.pack(side=tk.LEFT, padx=5, pady=5)

        self.fps_label = tk.Label(window, text="FPS: 0")
        self.fps_label.pack(side=tk.RIGHT, padx=5, pady=5)

        self.current_analysis = None
        self.is_running = True

        self.delay = 15
        self.prev_frame_time = 0
        self.update()

        self.window.mainloop()

    def start_horizontal(self):
        self.current_analysis = "horizontal"
        self.pface = horizontal.GooMedia(1080)

    def start_vertical(self):
        self.current_analysis = "vertical"
        self.pface = vertical.GooMedia(1080)

    def stop_analysis(self):
        self.current_analysis = None

    def update(self):
        if self.is_running:
            ret, frame = self.vid.read()
            if ret:
                # Calculate FPS
                current_time = time.time()
                fps = 1 / (current_time - self.prev_frame_time)
                self.prev_frame_time = current_time
                self.fps_label.config(text=f"FPS: {fps:.2f}")

                if self.current_analysis:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_resize, landmask = self.pface.extract(frame)
                    
                    if landmask:
                        all_points = self.pface.points_normalize(landmask, frame_resize.shape)
                        
                        if self.current_analysis == "horizontal":
                            points = horizontal.filter_points(all_points, horizontal.ALL_INDICES)
                            frame_resize = horizontal.process_frame(frame_resize, points)
                        else:
                            points = vertical.filter_points(all_points, vertical.ALL_INDICES)
                            frame_resize = vertical.process_frame(frame_resize, points)

                    self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame_resize))
                else:
                    self.photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
                
                self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

        self.window.after(self.delay, self.update)

    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()

# Add these functions to both angle_classification_optimized_H.py and angle_classification_optimized_V.py
def process_frame(frame, points):
    if points:
        max_y = max(point[1] for point in points)
        min_y = min(point[1] for point in points)
        max_x = max(point[0] for point in points)
        min_x = min(point[0] for point in points)

        mid_x = (max_x + min_x) // 2
        mid_y = (max_y + min_y) // 2

        left_hemisphere_count = sum(1 for point in points if point[0] < mid_x)
        right_hemisphere_count = len(points) - left_hemisphere_count

        area = (max_x - min_x) * (max_y - min_y)
        points_density = len(points) / area if area > 0 else 0
        left_hemisphere_density = left_hemisphere_count / (area/2) if area > 0 else 0
        right_hemisphere_density = right_hemisphere_count / (area/2) if area > 0 else 0

        left_hemisphere_ratio = left_hemisphere_density / points_density if points_density > 0 else 0
        right_hemisphere_ratio = right_hemisphere_density / points_density if points_density > 0 else 0

        check_angle = angle(left_hemisphere_ratio, right_hemisphere_ratio)

        for point in points:
            cv2.circle(frame, tuple(point), 2, (0, 255, 0), -1)

        cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)
        cv2.line(frame, (mid_x, min_y), (mid_x, max_y), (0, 0, 255), 2)

        cv2.putText(frame, check_angle, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    return frame

# Run the application
if __name__ == "__main__":
    FaceAngleApp(tk.Tk(), "Face Angle Analysis")