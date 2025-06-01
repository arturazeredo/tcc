import cv2
import mediapipe as mp
import numpy as np
import time
import tkinter as tk
from tkinter import messagebox  # Importação adicional
from PIL import Image, ImageTk

class LivenessDetectionApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)

        self.video_source = 1
        self.vid = cv2.VideoCapture(self.video_source)

        # Criação de canvases para as visualizações do usuário e do desenvolvedor
        self.user_canvas = tk.Canvas(window, width=640, height=480)
        self.user_canvas.pack(side=tk.LEFT)

        self.dev_canvas = tk.Canvas(window, width=640, height=480)
        self.dev_canvas.pack(side=tk.RIGHT)

        # Inicialização do MediaPipe
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Inicialização das variáveis
        self.calibration_done = False
        self.challenge_state = "start"
        self.challenge_start_time = 0
        self.score = 0
        self.total_score = 5  # Número total de testes
        self.blink_count = 0
        self.last_blink_ratio = 0
        self.blink_ratio = 0
        self.circle_position = None
        self.current_color = None
        self.base_sclera_color_left = None
        self.base_sclera_color_right = None
        self.calibration_frames = 0
        self.sclera_colors_left = []
        self.sclera_colors_right = []
        # Ajuste das cores para o formato BGR
        self.colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # Vermelho, Verde, Azul
        self.color_names = ["Vermelho", "Verde", "Azul"]
        self.color_index = 0
        self.color_variations_detected = [False, False, False]

        # Índices do face mesh
        self.LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
        self.LEFT_EYE = [33, 133]  # Para extração da cor da esclera
        self.RIGHT_EYE = [362, 263]
        self.LEFT_IRIS = [469, 470, 471, 472]
        self.RIGHT_IRIS = [474, 475, 476, 477]

        self.FACE_CENTER_INDICES = [1, 9, 152, 168, 8]  # Pontos centrais do rosto

        # Criação do botão Reiniciar
        self.btn_restart = tk.Button(window, text="Reiniciar", width=10, command=self.restart)
        self.btn_restart.pack()

        self.delay = 15
        self.update()

        self.window.mainloop()

    # Função para reiniciar o processo
    def restart(self):
        self.calibration_done = False
        self.challenge_state = "start"
        self.score = 0
        self.blink_count = 0
        self.calibration_frames = 0
        self.sclera_colors_left = []
        self.sclera_colors_right = []
        self.color_index = 0
        self.color_variations_detected = [False, False, False]
        self.current_color = None
        self.circle_position = None
        self.challenge_start_time = time.time()

    # Função principal de atualização
    def update(self):
        ret, frame = self.vid.read()
        if ret:
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)

            dev_frame = frame.copy()

            if results.multi_face_landmarks:
                if len(results.multi_face_landmarks) > 1:
                    self.display_message("Por favor, mostre apenas um rosto", frame)
                else:
                    face_landmarks = results.multi_face_landmarks[0]

                    # Verifica se o rosto está centralizado
                    if not self.is_face_centered(face_landmarks, frame):
                        self.display_message("Por favor, centralize seu rosto", frame)
                    else:
                        if not self.calibration_done:
                            self.calibrate_sclera_color(frame, face_landmarks)
                        else:
                            self.run_liveness_challenges(frame, face_landmarks)
            else:
                self.display_message("Nenhum rosto detectado", frame)

            # Exibe os frames
            self.display_user_frame(frame)
            self.display_dev_frame(dev_frame, results)

        self.window.after(self.delay, self.update)

    # Função para calibrar a cor da esclera
    def calibrate_sclera_color(self, frame, landmarks):
        self.calibration_frames += 1
        left_sclera_color = self.get_sclera_color(frame, landmarks, self.LEFT_EYE, self.LEFT_IRIS)
        right_sclera_color = self.get_sclera_color(frame, landmarks, self.RIGHT_EYE, self.RIGHT_IRIS)
        self.sclera_colors_left.append(left_sclera_color)
        self.sclera_colors_right.append(right_sclera_color)

        if self.calibration_frames >= 30:
            self.base_sclera_color_left = np.mean(self.sclera_colors_left, axis=0)
            self.base_sclera_color_right = np.mean(self.sclera_colors_right, axis=0)
            self.calibration_done = True
            self.challenge_state = "start"
            self.challenge_start_time = time.time()
        else:
            self.display_message(f"Calibrando... {self.calibration_frames}/30", frame)

    # Função para obter a cor da esclera
    def get_sclera_color(self, frame, landmarks, eye_indices, iris_indices):
        height, width = frame.shape[:2]
        eye_region = np.array(
            [(int(landmarks.landmark[idx].x * width), int(landmarks.landmark[idx].y * height)) for idx in eye_indices],
            dtype=np.int32
        )
        iris_region = np.array(
            [(int(landmarks.landmark[idx].x * width), int(landmarks.landmark[idx].y * height)) for idx in iris_indices],
            dtype=np.int32
        )

        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [eye_region], 255)
        cv2.fillPoly(mask, [iris_region], 0)

        sclera_color = cv2.mean(frame, mask=mask)[:3]
        return sclera_color

    # Função para verificar se o rosto está centralizado
    def is_face_centered(self, landmarks, frame):
        img_h, img_w = frame.shape[:2]
        face_coords = np.array(
            [(int(landmarks.landmark[idx].x * img_w), int(landmarks.landmark[idx].y * img_h)) for idx in self.FACE_CENTER_INDICES]
        )
        x_coords = face_coords[:, 0]
        y_coords = face_coords[:, 1]
        face_center_x = np.mean(x_coords)
        face_center_y = np.mean(y_coords)

        if abs(face_center_x - img_w / 2) < img_w * 0.1 and abs(face_center_y - img_h / 2) < img_h * 0.1:
            return True
        else:
            return False

    # Função para executar os desafios de liveness
    def run_liveness_challenges(self, frame, landmarks):
        if self.challenge_state == "start":
            self.challenge_state = "look_left"
            self.challenge_start_time = time.time()
            self.circle_position = (50, frame.shape[0] // 2)  # Lado esquerdo da tela
        elif self.challenge_state == "look_left":
            self.look_challenge(frame, landmarks, "left")
        elif self.challenge_state == "look_right":
            self.look_challenge(frame, landmarks, "right")
        elif self.challenge_state == "color_change":
            self.color_change_challenge(frame, landmarks)
        elif self.challenge_state == "blink":
            self.blink_challenge(frame, landmarks)
        elif self.challenge_state == "end":
            self.end_challenges()

    # Função para o desafio de olhar para o círculo
    def look_challenge(self, frame, landmarks, direction):
        duration = 2  # Duração do desafio em segundos
        elapsed_time = time.time() - self.challenge_start_time
        if elapsed_time > duration:
            iris_position = self.get_iris_position(landmarks)
            if (direction == "left" and iris_position == "Left") or (direction == "right" and iris_position == "Right"):
                self.score += 1
            else:
                pass  # Teste falhado, não incrementa o score
            if direction == "left":
                self.challenge_state = "look_right"
                self.circle_position = (frame.shape[1] - 50, frame.shape[0] // 2)  # Lado direito da tela
            else:
                self.challenge_state = "color_change"
                self.color_index = 0
            self.challenge_start_time = time.time()
        else:
            direction_text = "esquerda" if direction == "left" else "direita"
            self.display_message(f"Olhe para o círculo à {direction_text}", frame)
            cv2.circle(frame, self.circle_position, 20, (0, 255, 0), -1)

    # Função para obter a posição da íris
    def get_iris_position(self, landmarks):
        left_iris = np.mean(
            [(landmarks.landmark[idx].x, landmarks.landmark[idx].y) for idx in self.LEFT_IRIS], axis=0
        )
        right_iris = np.mean(
            [(landmarks.landmark[idx].x, landmarks.landmark[idx].y) for idx in self.RIGHT_IRIS], axis=0
        )
        iris_x = (left_iris[0] + right_iris[0]) / 2

        if iris_x < 0.45:
            return "Left"
        elif iris_x > 0.55:
            return "Right"
        else:
            return "Center"

    # Função para o desafio de mudança de cor na esclera
    def color_change_challenge(self, frame, landmarks):
        duration_per_color = 2  # Duração para exibir cada cor
        total_colors = len(self.colors)
        elapsed_time = time.time() - self.challenge_start_time

        if self.color_index >= total_colors:
            # Desafio concluído
            # Conta o número de variações de cor detectadas
            num_variations_detected = sum(self.color_variations_detected)
            self.score += num_variations_detected
            self.challenge_state = "blink"
            self.challenge_start_time = time.time()
        else:
            if elapsed_time > duration_per_color:
                # Avança para a próxima cor
                self.challenge_start_time = time.time()
                self.color_index +=1
            else:
                # Exibe a cor atual
                color = self.colors[self.color_index]
                overlay = np.full(frame.shape, color, dtype=np.uint8)
                frame[:] = cv2.addWeighted(frame, 0.5, overlay, 0.5, 0)
                self.display_message(f"Mantenha olhando - Cor: {self.color_names[self.color_index]}", frame)

                # Verifica a variação de cor na esclera
                left_sclera_color = self.get_sclera_color(frame, landmarks, self.LEFT_EYE, self.LEFT_IRIS)
                right_sclera_color = self.get_sclera_color(frame, landmarks, self.RIGHT_EYE, self.RIGHT_IRIS)

                left_color_diff = np.abs(left_sclera_color - self.base_sclera_color_left)
                right_color_diff = np.abs(right_sclera_color - self.base_sclera_color_right)

                if np.any(left_color_diff > 3) or np.any(right_color_diff > 3):
                    self.color_variations_detected[self.color_index] = True

    # Função para o desafio de piscar
    def blink_challenge(self, frame, landmarks):
        duration = 3  # Duração do desafio em segundos
        elapsed_time = time.time() - self.challenge_start_time
        if elapsed_time > duration:
            if self.blink_count > 0:
                self.score += 1
            self.challenge_state = "end"
        else:
            self.display_message("Pisque naturalmente", frame)
            blink_ratio = self.calculate_blink_ratio(landmarks)
            if blink_ratio < 0.25 and self.last_blink_ratio >= 0.25:
                self.blink_count += 1
            self.last_blink_ratio = blink_ratio

    # Função para calcular a razão de piscada
    def calculate_blink_ratio(self, landmarks):
        left_eye_indices = self.LEFT_EYE_INDICES
        right_eye_indices = self.RIGHT_EYE_INDICES
        left_eye = [landmarks.landmark[idx] for idx in left_eye_indices]
        right_eye = [landmarks.landmark[idx] for idx in right_eye_indices]

        left_ratio = self.eye_aspect_ratio(left_eye)
        right_ratio = self.eye_aspect_ratio(right_eye)

        self.blink_ratio = (left_ratio + right_ratio) / 2  # Armazena o blink ratio para exibição
        return self.blink_ratio

    # Função para calcular o EAR
    def eye_aspect_ratio(self, eye):
        # eye é uma lista de 6 landmarks
        # Distâncias verticais
        p2_p6 = self.euclidean_distance(eye[1], eye[5])
        p3_p5 = self.euclidean_distance(eye[2], eye[4])
        # Distância horizontal
        p1_p4 = self.euclidean_distance(eye[0], eye[3])
        if p1_p4 == 0:
            return 0  # Evita divisão por zero
        return (p2_p6 + p3_p5) / (2.0 * p1_p4)

    # Função para calcular a distância euclidiana
    def euclidean_distance(self, p1, p2):
        x1, y1 = p1.x, p1.y
        x2, y2 = p2.x, p2.y
        return np.sqrt((x2 - x1) ** 2 + (y2 - y1) **2)

    # Função para concluir os desafios
    def end_challenges(self):
        success_rate = (self.score / self.total_score) * 100 if self.total_score > 0 else 0
        failed_tests = []

        # Verificação dos testes falhados
        if not self.color_variations_detected[0]:
            failed_tests.append("Variação de cor Vermelho")
        if not self.color_variations_detected[1]:
            failed_tests.append("Variação de cor Verde")
        if not self.color_variations_detected[2]:
            failed_tests.append("Variação de cor Azul")
        if self.blink_count == 0:
            failed_tests.append("Piscar")
        if self.score < self.total_score - (3 - sum(self.color_variations_detected)):
            failed_tests.append("Olhar para a esquerda e/ou direita")

        # Preparação da mensagem de resultado
        if success_rate == 100:
            result_message = "Parabéns! Você passou em todos os testes de liveness.\n"
        else:
            result_message = "Teste concluído.\n"

        result_message += f"Pontuação: {self.score}/{self.total_score} ({success_rate:.1f}%)\n"

        if failed_tests:
            result_message += "Testes falhados:\n"
            for test in failed_tests:
                result_message += f" - {test}\n"
        else:
            result_message += "Todos os testes foram bem-sucedidos!"

        # Exibe mensagem final em um popup
        messagebox.showinfo("Resultado do Liveness", result_message)

        # Reinicia após mostrar os resultados
        self.restart()

    # Função para exibir mensagens na tela do usuário
    def display_message(self, message, frame):
        if frame is not None:
            cv2.putText(frame, message, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Função para exibir o frame na tela do usuário
    def display_user_frame(self, frame):
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)
        self.user_canvas.imgtk = imgtk
        self.user_canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)

    # Função para exibir o frame na tela do desenvolvedor
    def display_dev_frame(self, frame, results):
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                self.draw_face_landmarks(frame, face_landmarks)

        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)
        self.dev_canvas.imgtk = imgtk
        self.dev_canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)

        # Exibe informações adicionais de debug
        debug_info = f"Desafio: {self.challenge_state}\n"
        debug_info += f"Score: {self.score}/{self.total_score}\n"
        debug_info += f"Piscadas: {self.blink_count}\n"
        debug_info += f"Blink Ratio: {self.blink_ratio:.3f}\n"
        self.dev_canvas.create_text(10, 30, anchor=tk.NW, text=debug_info, fill="red", font=("Arial", 12))

    # Função para desenhar os pontos faciais
    def draw_face_landmarks(self, frame, landmarks):
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        mp_drawing.draw_landmarks(
            image=frame,
            landmark_list=landmarks,
            connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
        )

    # Função para liberar os recursos
    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()

# Cria a janela e inicia a aplicação
root = tk.Tk()
LivenessDetectionApp(root, "Liveness Detection")