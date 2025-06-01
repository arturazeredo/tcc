from src.mediapipe import GooMedia
import cv2
import numpy as np
from src.visualize import *
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from src.utils import image_resize
from PIL import Image
import mediapipe as mp
import json
import os

#%matplotlib qt
pface = GooMedia(1080) # Extrator dos pontos

# Defina o caminho para a pasta de imagens
image_dir = r"images\inputs\artur\36points"

# Crie um dicionário vazio para armazenar os resultados
result = {}

# Listas de pontos para os olhos e íris
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

# Loop pelas imagens na pasta
for filename in os.listdir(image_dir):
    # Verifique se o arquivo é uma imagem
    if filename.endswith(".jpg") or filename.endswith(".png"):
        # Leia a imagem
        image_path = os.path.join(image_dir, filename)
        image = cv2.imread(image_path)
        
        # Resize image
        image_r = cv2.resize(image, (480, 640)) 
        image_resize , landmask = pface.extract(image_r)
        points = pface.points_normalize(landmask, image_r.shape)

        points_face = draw_face_one_color(image_resize, points)

        dicionario = {}  # Reinicialize o dicionario a cada iteração
        dicionario = {i+1: coordenada for i, coordenada in enumerate(points)}

        # Filtrar as chaves do dicionário para os valores nas listas
        keys_to_keep = []
        for eye_or_iris in [LEFT_EYE, RIGHT_EYE, LEFT_IRIS, RIGHT_IRIS]:
            for key in dicionario:
                if key in eye_or_iris:
                    keys_to_keep.append(key)

        # Criar um novo dicionário com as chaves filtradas
        filtered_dict = {key: dicionario[key] for key in keys_to_keep}

        # Adicione uma linha ao JSON correspondente ao dicionário daquele arquivo de imagem
        with open('result.json', 'a') as f:
            json.dump({filename: filtered_dict}, f)
            f.write('\n')