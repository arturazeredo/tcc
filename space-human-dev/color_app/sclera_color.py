import cv2
import mediapipe as mp
import numpy as np
import time
from screeninfo import get_monitors

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(1)

LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # Vermelho, Verde, Azul
color_names = ["Vermelho", "Verde", "Azul"]

current_color_index = 0
color_change_interval = 2  # segundos

# Variáveis para controle de perda de detecção
last_detection_time = time.time()
detection_timeout = 1.0  # segundos

def get_screen_resolution():
    monitors = get_monitors()
    if monitors:
        main_monitor = monitors[0]
        return main_monitor.width, main_monitor.height
    return None

def create_eye_mask(frame, landmarks, eye_indices, iris_indices):
    height, width = frame.shape[:2]
    eye_region = np.array([(landmarks[idx].x * width, landmarks[idx].y * height) for idx in eye_indices], dtype=np.int32)
    iris_region = np.array([(landmarks[idx].x * width, landmarks[idx].y * height) for idx in iris_indices], dtype=np.int32)
    
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [eye_region], 255)
    cv2.fillPoly(mask, [iris_region], 0)
    
    return mask, eye_region

def get_sclera_color(frame, landmarks, eye_indices, iris_indices):
    mask, _ = create_eye_mask(frame, landmarks, eye_indices, iris_indices)
    sclera_color = cv2.mean(frame, mask=mask)[:3]
    return sclera_color

def put_text_with_background(img, text, pos, font, font_scale, text_color, bg_color):
    x, y = pos
    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness=1)
    cv2.rectangle(img, (x, y - text_height - 5), (x + text_width, y + 5), bg_color, -1)
    cv2.putText(img, text, (x, y), font, font_scale, text_color, thickness=1)

def calibrate_sclera_color():
    calibration_frames = 120
    left_sclera_colors = []
    right_sclera_colors = []

    for _ in range(calibration_frames):
        success, image = cap.read()
        if not success:
            continue

        image = cv2.flip(image, 1)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_image)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks = face_landmarks.landmark
                left_sclera_colors.append(get_sclera_color(image, landmarks, LEFT_EYE, LEFT_IRIS))
                right_sclera_colors.append(get_sclera_color(image, landmarks, RIGHT_EYE, RIGHT_IRIS))

        put_text_with_background(image, "Calibrando... Olhe diretamente para a câmera", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), (0, 0, 0))
        cv2.imshow('Calibração', image)
        cv2.waitKey(1)

    left_base_color = np.mean(left_sclera_colors, axis=0)
    right_base_color = np.mean(right_sclera_colors, axis=0)
    return left_base_color, right_base_color

def get_eye_crop(image, eye_region, padding=10):
    x, y, w, h = cv2.boundingRect(eye_region)
    return image[max(0, y-padding):min(image.shape[0], y+h+padding), max(0, x-padding):min(image.shape[1], x+w+padding)]

# Calibração inicial
left_base_color, right_base_color = calibrate_sclera_color()
print(f"Cor base da esclera esquerda: {left_base_color}")
print(f"Cor base da esclera direita: {right_base_color}")

last_color_change = time.time()

# Variáveis para armazenar os últimos recortes válidos dos olhos
last_valid_left_eye = None
last_valid_right_eye = None

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Falha ao capturar o frame.")
        continue

    image = cv2.flip(image, 1)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_image)

    height, width = image.shape[:2]

    resolution = get_screen_resolution()
    
    color_screen = np.zeros((resolution[1], resolution[0], 3), dtype=np.uint8)
    eye_zoom_screen = np.zeros((300, 600, 3), dtype=np.uint8)

    face_detected = False
    if results.multi_face_landmarks:
        face_detected = True
        last_detection_time = time.time()
        for face_landmarks in results.multi_face_landmarks:
            landmarks = face_landmarks.landmark

            left_mask, left_eye_region = create_eye_mask(image, landmarks, LEFT_EYE, LEFT_IRIS)
            right_mask, right_eye_region = create_eye_mask(image, landmarks, RIGHT_EYE, RIGHT_IRIS)

            left_sclera_color = get_sclera_color(image, landmarks, LEFT_EYE, LEFT_IRIS)
            right_sclera_color = get_sclera_color(image, landmarks, RIGHT_EYE, RIGHT_IRIS)

            left_color_diff = np.subtract(left_sclera_color, left_base_color)
            right_color_diff = np.subtract(right_sclera_color, right_base_color)

            # Obter recortes dos olhos
            left_eye_crop = get_eye_crop(image, left_eye_region)
            right_eye_crop = get_eye_crop(image, right_eye_region)

            # Atualizar os últimos recortes válidos
            last_valid_left_eye = left_eye_crop
            last_valid_right_eye = right_eye_crop

    # Se não houver detecção, use os últimos recortes válidos
    if not face_detected and time.time() - last_detection_time < detection_timeout:
        left_eye_crop = last_valid_left_eye
        right_eye_crop = last_valid_right_eye
    elif not face_detected:
        left_eye_crop = np.zeros((100, 100, 3), dtype=np.uint8)
        right_eye_crop = np.zeros((100, 100, 3), dtype=np.uint8)
        put_text_with_background(eye_zoom_screen, "Rosto não detectado", (150, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), (0, 0, 0))

    # Redimensionar os recortes para exibição
    if left_eye_crop is not None and right_eye_crop is not None:
        display_height = 250
        display_width = 250
        left_eye_resized = cv2.resize(left_eye_crop, (display_width, display_height))
        right_eye_resized = cv2.resize(right_eye_crop, (display_width, display_height))

        # Exibir os recortes dos olhos na tela de zoom com espaço entre eles
        eye_zoom_screen[:display_height, :display_width] = left_eye_resized
        eye_zoom_screen[:display_height, -display_width:] = right_eye_resized

        # Adicionar legendas e informações de diferença de cor aos olhos
        if face_detected:
            left_color_text = f"Dif: ({left_color_diff[0]:.0f}, {left_color_diff[1]:.0f}, {left_color_diff[2]:.0f})"
            right_color_text = f"Dif: ({right_color_diff[0]:.0f}, {right_color_diff[1]:.0f}, {right_color_diff[2]:.0f})"
            
            put_text_with_background(eye_zoom_screen, "Olho Esquerdo", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), (0, 0, 0))
            put_text_with_background(eye_zoom_screen, left_color_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), (0, 0, 0))
            
            put_text_with_background(eye_zoom_screen, "Olho Direito", (eye_zoom_screen.shape[1] - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), (0, 0, 0))
            put_text_with_background(eye_zoom_screen, right_color_text, (eye_zoom_screen.shape[1] - 200, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), (0, 0, 0))

    current_time = time.time()
    if current_time - last_color_change > color_change_interval:
        current_color_index = (current_color_index + 1) % len(colors)
        last_color_change = current_time

    color_overlay = np.full(color_screen.shape, colors[current_color_index], dtype=np.uint8)
    color_screen = cv2.addWeighted(color_screen, 0.2, color_overlay, 0.8, 0)

    put_text_with_background(color_screen, f"Cor atual: {color_names[current_color_index]}", (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), (0, 0, 0))

    cv2.imshow('Câmera', image)
    cv2.imshow('Cores e Valores RGB', color_screen)
    cv2.imshow('Zoom dos Olhos', eye_zoom_screen)

    if cv2.waitKey(5) & 0xFF == 27 or cv2.waitKey(5) & 0xFF == ord('q') or cv2.waitKey(5) & 0xFF == ord('Q'):  # Pressione ESC para sair
        break

cap.release()
cv2.destroyAllWindows()