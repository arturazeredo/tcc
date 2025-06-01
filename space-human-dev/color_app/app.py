from flask import Flask, render_template, Response, jsonify, request, send_file, send_from_directory
import cv2
import mediapipe as mp
import numpy as np
import time
import threading
import os
import zipfile
import io

app = Flask(__name__)
output_folder = 'captured_images'

# Configurações globais
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

colors = [(0, 255, 255), (255, 255, 0), (255, 0, 255)]  # Ciano, Amarelo, Magenta
color_names = ["Amarelo", "Ciano", "Magenta"]

# Variáveis globais
camera = cv2.VideoCapture(0)
current_color_index = 0
capture_interval = 2  # segundos
capturing = False
images_to_capture = 0
images_captured = 0
output_folder = "captured_images"
calibration_complete = False
left_base_color = None
right_base_color = None
calibration_frames = 0
calibration_threshold = 30  # Número de frames para considerar a calibração completa

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

def create_eye_mask(frame, landmarks, eye_indices, iris_indices):
    height, width = frame.shape[:2]
    eye_region = np.array([(landmarks[idx].x * width, landmarks[idx].y * height) for idx in eye_indices], dtype=np.int32)
    iris_region = np.array([(landmarks[idx].x * width, landmarks[idx].y * height) for idx in iris_indices], dtype=np.int32)
    
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [eye_region], 255)
    cv2.fillPoly(mask, [iris_region], 0)
    
    return mask

def get_sclera_color(frame, landmarks, eye_indices, iris_indices):
    mask = create_eye_mask(frame, landmarks, eye_indices, iris_indices)
    sclera_color = cv2.mean(frame, mask=mask)[:3]
    return sclera_color

def put_text_with_background(img, text, pos, font, font_scale, text_color, bg_color):
    x, y = pos
    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness=1)
    cv2.rectangle(img, (x, y - text_height - 5), (x + text_width, y + 5), bg_color, -1)
    cv2.putText(img, text, (x, y), font, font_scale, text_color, thickness=1)

def calibrate_sclera_color():
    global left_base_color, right_base_color, calibration_complete
    calibration_frames = 60
    left_sclera_colors = []
    right_sclera_colors = []

    for _ in range(calibration_frames):
        success, image = camera.read()
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
        ret, buffer = cv2.imencode('.jpg', image)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    left_base_color = np.mean(left_sclera_colors, axis=0)
    right_base_color = np.mean(right_sclera_colors, axis=0)
    calibration_complete = True

def generate_frames():
    global current_color_index, capturing, images_captured, images_to_capture, calibration_complete, calibration_frames

    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    landmarks = face_landmarks.landmark
                    left_sclera_color = get_sclera_color(frame, landmarks, LEFT_EYE, LEFT_IRIS)
                    right_sclera_color = get_sclera_color(frame, landmarks, RIGHT_EYE, RIGHT_IRIS)

                    if not calibration_complete:
                        calibration_frames += 1
                        if calibration_frames >= calibration_threshold:
                            calibration_complete = True
                            put_text_with_background(frame, "Calibração concluída!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), (0, 0, 0))
                        else:
                            put_text_with_background(frame, "Calibrando...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), (0, 0, 0))
                    elif capturing and images_captured < images_to_capture:
                        color = colors[current_color_index]
                        cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), color, -1)
                        cv2.addWeighted(frame, 0.5, cv2.rectangle(frame.copy(), (0, 0), (frame.shape[1], frame.shape[0]), color, -1), 0.5, 0, frame)
                        
                        put_text_with_background(frame, f"Olhe para a cor {color_names[current_color_index]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), (0, 0, 0))
                        put_text_with_background(frame, f"Imagem {images_captured + 1} de {images_to_capture}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), (0, 0, 0))

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def capture_images():
    global capturing, current_color_index, images_captured

    while capturing and images_captured < images_to_capture:
        time.sleep(2)  # Aguarda 2 segundos antes de capturar cada imagem
        success, image = camera.read()
        if success:
            filename = f"{output_folder}/image_{color_names[current_color_index]}_{images_captured+1}.jpg"
            cv2.imwrite(filename, image)
            images_captured += 1
            current_color_index = (current_color_index + 1) % len(colors)
            time.sleep(capture_interval - 2)  # Subtrai 2 segundos do intervalo total
    
    capturing = False

@app.route('/get_captured_images')
def get_captured_images():
    images = [f for f in os.listdir(output_folder) if f.endswith('.jpg')]
    return jsonify({'images': images})

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_capture', methods=['POST'])
def start_capture():
    global capturing, images_to_capture, images_captured
    if not capturing and calibration_complete:
        images_to_capture = int(request.form['count'])
        images_captured = 0
        capturing = True
        threading.Thread(target=capture_images).start()
    return jsonify({'status': 'started'})

@app.route('/stop_capture', methods=['POST'])
def stop_capture():
    global capturing
    capturing = False
    return jsonify({'status': 'stopped'})

@app.route('/get_capture_status')
def get_capture_status():
    global capturing, images_captured, images_to_capture, calibration_complete
    return jsonify({
        'capturing': capturing,
        'images_captured': images_captured,
        'images_to_capture': images_to_capture,
        'current_color': color_names[current_color_index],
        'calibration_complete': calibration_complete
    })

@app.route('/calibrate')
def calibrate():
    return Response(calibrate_sclera_color(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/list_captured_images')
def list_captured_images():
    images = [f for f in os.listdir(output_folder) if f.endswith('.jpg')]
    return jsonify({'images': images})

@app.route('/download_images')
def download_images():
    memory_file = io.BytesIO()
    with zipfile.ZipFile(memory_file, 'w') as zf:
        for root, dirs, files in os.walk(output_folder):
            for file in files:
                zf.write(os.path.join(root, file), file)
    memory_file.seek(0)
    return send_file(memory_file, attachment_filename='captured_images.zip', as_attachment=True)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/captured_images/<path:filename>')
def serve_image(filename):
    return send_from_directory(output_folder, filename)

if __name__ == '__main__':
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    app.run(debug=True)