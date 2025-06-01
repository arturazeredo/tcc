import cv2
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import time
import numpy as np
import mediapipe as mp
import utils

# Import the necessary functions from other modules, not the entire modules
from main import LEFT_IRIS, RIGHT_IRIS
from landmarks_detection import FACE_OVAL, LEFT_EYE, RIGHT_EYE, LEFT_EYEBROW, RIGHT_EYEBROW, LIPS
from blink import blinkingRatio, FONTS

class CameraApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)

        self.video_source = 1 # 0 for default camera, 1 for external camera
        self.vid = cv2.VideoCapture(self.video_source)

        self.canvas = tk.Canvas(window, width=self.vid.get(cv2.CAP_PROP_FRAME_WIDTH), height=self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.canvas.pack()

        self.btn_frame = ttk.Frame(window)
        self.btn_frame.pack(pady=10)

        self.btn_main = ttk.Button(self.btn_frame, text="Iris Detection", command=lambda: self.activate_algorithm("main"))
        self.btn_main.grid(row=0, column=0, padx=5)

        self.btn_landmarks = ttk.Button(self.btn_frame, text="Landmarks Detection", command=lambda: self.activate_algorithm("landmarks"))
        self.btn_landmarks.grid(row=0, column=1, padx=5)

        self.btn_blink = ttk.Button(self.btn_frame, text="Blink Detection", command=lambda: self.activate_algorithm("blink"))
        self.btn_blink.grid(row=0, column=2, padx=5)

        self.active_algorithm = None
        self.is_running = True
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()

        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Blink detection variables
        self.CEF_COUNTER = 0
        self.TOTAL_BLINKS = 0
        self.CLOSED_EYES_FRAME = 3

        self.update()

        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)

    def activate_algorithm(self, algorithm):
        self.active_algorithm = algorithm
        self.reset_button_states()
        if algorithm == "main":
            self.btn_main.state(['pressed'])
        elif algorithm == "landmarks":
            self.btn_landmarks.state(['pressed'])
        elif algorithm == "blink":
            self.btn_blink.state(['pressed'])

    def reset_button_states(self):
        self.btn_main.state(['!pressed'])
        self.btn_landmarks.state(['!pressed'])
        self.btn_blink.state(['!pressed'])

    def update(self):
        ret, frame = self.vid.read()
        if ret:
            frame = cv2.flip(frame, 1)

            if self.active_algorithm == "main":
                frame = self.process_main(frame)
            elif self.active_algorithm == "landmarks":
                frame = self.process_landmarks(frame)
            elif self.active_algorithm == "blink":
                frame = self.process_blink(frame)

            self.frame_count += 1
            elapsed_time = time.time() - self.start_time
            if elapsed_time > 1:
                self.fps = self.frame_count / elapsed_time
                self.frame_count = 0
                self.start_time = time.time()

            cv2.putText(frame, f"FPS: {self.fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            self.photo = self.convert_frame_to_photo(frame)
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

        if self.is_running:
            self.window.after(15, self.update)

    def process_main(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        if results.multi_face_landmarks:
            mesh_points = np.array([np.multiply([p.x, p.y], [frame.shape[1], frame.shape[0]]).astype(int) for p in results.multi_face_landmarks[0].landmark])
            (l_cx, l_cy), l_radius = cv2.minEnclosingCircle(mesh_points[LEFT_IRIS])
            (r_cx, r_cy), r_radius = cv2.minEnclosingCircle(mesh_points[RIGHT_IRIS])
            center_left = np.array([l_cx, l_cy], dtype=np.int32)
            center_right = np.array([r_cx, r_cy], dtype=np.int32)
            cv2.circle(frame, center_left, int(l_radius), (255, 0, 255), 1, cv2.LINE_AA)
            cv2.circle(frame, center_right, int(r_radius), (255, 0, 255), 1, cv2.LINE_AA)
        return frame

    def process_landmarks(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        if results.multi_face_landmarks:
            mesh_coords = self.landmarksDetection(frame, results, True)
            frame = utils.fillPolyTrans(frame, [mesh_coords[p] for p in FACE_OVAL], utils.GRAY, opacity=0.6)
            frame = utils.fillPolyTrans(frame, [mesh_coords[p] for p in LEFT_EYE], utils.GREEN, opacity=0.4)
            frame = utils.fillPolyTrans(frame, [mesh_coords[p] for p in RIGHT_EYE], utils.GREEN, opacity=0.4)
            frame = utils.fillPolyTrans(frame, [mesh_coords[p] for p in LEFT_EYEBROW], utils.ORANGE, opacity=0.4)
            frame = utils.fillPolyTrans(frame, [mesh_coords[p] for p in RIGHT_EYEBROW], utils.ORANGE, opacity=0.4)
            frame = utils.fillPolyTrans(frame, [mesh_coords[p] for p in LIPS], utils.BLACK, opacity=0.3)
        return frame

    def process_blink(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        if results.multi_face_landmarks:
            mesh_coords = self.landmarksDetection(frame, results, False)
            try:
                ratio = blinkingRatio(frame, mesh_coords, RIGHT_EYE, LEFT_EYE)
                cv2.putText(frame, f'Ratio: {round(ratio,2)}', (50, 100), FONTS, 1.0, utils.GREEN, 2)

                if ratio > 5.5:
                    self.CEF_COUNTER += 1
                    cv2.putText(frame, 'Blink', (200, 50), FONTS, 1.3, utils.PINK, 2)
                else:
                    if self.CEF_COUNTER > self.CLOSED_EYES_FRAME:
                        self.TOTAL_BLINKS += 1
                        self.CEF_COUNTER = 0

                cv2.putText(frame, f'Total Blinks: {self.TOTAL_BLINKS}', (50, 150), FONTS, 1.0, utils.GREEN, 2)

            except ZeroDivisionError:
                pass  # Ignore the frame if a division by zero occurs

        return frame

    def landmarksDetection(self, img, results, draw=False):
        img_height, img_width = img.shape[:2]
        mesh_coord = [(int(point.x * img_width), int(point.y * img_height)) for point in results.multi_face_landmarks[0].landmark]
        if draw:
            [cv2.circle(img, p, 2, utils.GREEN, -1) for p in mesh_coord]
        return mesh_coord

    def convert_frame_to_photo(self, frame):
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        return ImageTk.PhotoImage(image=pil_image)

    def on_closing(self):
        self.is_running = False
        self.vid.release()
        self.window.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = CameraApp(root, "Camera App")
    root.mainloop()