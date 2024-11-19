from flask import Flask, request, render_template
from PIL import Image, ImageEnhance, ImageOps
import numpy as np
import json
import subprocess
import mediapipe as mp
import cv2
import matplotlib.pyplot as plt
import base64
import io

# Configurar Matplotlib para que use el backend 'Agg'
import matplotlib
matplotlib.use('Agg')

app = Flask(__name__)

# Inicializar MediaPipe para la detección de rostros
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Función para seleccionar 15 puntos clave relevantes
def extract_key_points(landmarks, image_shape):
    selected_indices = {
        'left_eye_center': 159,
        'right_eye_center': 386,
        'left_eye_inner_corner': 133,
        'left_eye_outer_corner': 33,
        'right_eye_inner_corner': 362,
        'right_eye_outer_corner': 263,
        'left_eyebrow_inner_end': 70,
        'left_eyebrow_outer_end': 105,
        'right_eyebrow_inner_end': 336,
        'right_eyebrow_outer_end': 334,
        'nose_tip': 1,
        'mouth_left_corner': 61,
        'mouth_right_corner': 291,
        'mouth_center_top_lip': 13,
        'mouth_center_bottom_lip': 14
    }
    
    key_points = {}
    for key, index in selected_indices.items():
        x = int(landmarks[index].x * image_shape[1])
        y = int(landmarks[index].y * image_shape[0])
        key_points[key] = (x, y)
    return key_points

def puntos(img, img_str):
    # Detectar rostros y obtener puntos clave usando MediaPipe
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        if not results.multi_face_landmarks:
            return 'No se detectaron rostros en la imagen.'

        # Tomar el primer rostro detectado
        face_landmarks = results.multi_face_landmarks[0]

        # Obtener los puntos clave faciales
        key_points = extract_key_points(face_landmarks.landmark, img.shape)

    # Guardar las coordenadas en un diccionario
    coordinates = {
        'left_eye_center_x': key_points['left_eye_center'][0],
        'left_eye_center_y': key_points['left_eye_center'][1],
        'right_eye_center_x': key_points['right_eye_center'][0],
        'right_eye_center_y': key_points['right_eye_center'][1],
        'left_eye_inner_corner_x': key_points['left_eye_inner_corner'][0],
        'left_eye_inner_corner_y': key_points['left_eye_inner_corner'][1],
        'left_eye_outer_corner_x': key_points['left_eye_outer_corner'][0],
        'left_eye_outer_corner_y': key_points['left_eye_outer_corner'][1],
        'right_eye_inner_corner_x': key_points['right_eye_inner_corner'][0],
        'right_eye_inner_corner_y': key_points['right_eye_inner_corner'][1],
        'right_eye_outer_corner_x': key_points['right_eye_outer_corner'][0],
        'right_eye_outer_corner_y': key_points['right_eye_outer_corner'][1],
        'left_eyebrow_inner_end_x': key_points['left_eyebrow_inner_end'][0],
        'left_eyebrow_inner_end_y': key_points['left_eyebrow_inner_end'][1],
        'left_eyebrow_outer_end_x': key_points['left_eyebrow_outer_end'][0],
        'left_eyebrow_outer_end_y': key_points['left_eyebrow_outer_end'][1],
        'right_eyebrow_inner_end_x': key_points['right_eyebrow_inner_end'][0],
        'right_eyebrow_inner_end_y': key_points['right_eyebrow_inner_end'][1],
        'right_eyebrow_outer_end_x': key_points['right_eyebrow_outer_end'][0],
        'right_eyebrow_outer_end_y': key_points['right_eyebrow_outer_end'][1],
        'nose_tip_x': key_points['nose_tip'][0],
        'nose_tip_y': key_points['nose_tip'][1],
        'mouth_left_corner_x': key_points['mouth_left_corner'][0],
        'mouth_left_corner_y': key_points['mouth_left_corner'][1],
        'mouth_right_corner_x': key_points['mouth_right_corner'][0],
        'mouth_right_corner_y': key_points['mouth_right_corner'][1],
        'mouth_center_top_lip_x': key_points['mouth_center_top_lip'][0],
        'mouth_center_top_lip_y': key_points['mouth_center_top_lip'][1],
        'mouth_center_bottom_lip_x': key_points['mouth_center_bottom_lip'][0],
        'mouth_center_bottom_lip_y': key_points['mouth_center_bottom_lip'][1],
        'image': img_str
    }
    return coordinates, key_points

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route('/procesar', methods=['POST'])
def procesar():
    images_data = {}
    try:
        # Verificar si el archivo fue incluido en la solicitud
        if 'file' not in request.files:
            return 'No se ha proporcionado ningún archivo.'

        file = request.files['file']

        # Verificar si el archivo tiene un nombre válido
        if file.filename == '':
            return 'No se ha seleccionado ningún archivo.'

        # Leer y abrir la imagen
        img = Image.open(file)
        img = img.resize((96, 96))  # Redimensionar la imagen a 96x96
        img = img.convert('L')  # Convertir a escala de grises

        # Convertir la imagen a un array para detección de puntos clave
        img_np = np.array(img)
        img_str = img_np.flatten().tolist()  # Convertir a lista plana

        # Obtener los puntos clave de la imagen
        key_points, key_points2 = puntos(img_np, img_str)

        # Dibujar las cruces rojas en los puntos clave
        for point in key_points2.values():
            cv2.drawMarker(img_np, point, color=(0, 0, 255), markerType=cv2.MARKER_CROSS, 
                           markerSize=3, thickness=1, line_type=cv2.LINE_AA)

        # Convertir la imagen con puntos clave a PIL
        img_with_points = Image.fromarray(img_np)

        # Efecto 1: Aumentar el brillo
        enhancer = ImageEnhance.Brightness(img_with_points)
        img_brightness = enhancer.enhance(1.5)

        # Efecto 2: Voltear horizontalmente
        img_flip_horizontal = ImageOps.mirror(img_with_points)

        # Efecto 3: Voltear de cabeza
        img_flip_vertical = ImageOps.flip(img_with_points)

        # Convertir cada imagen a base64 para el template
        for key, processed_img in zip(
            ['original', 'brillo', 'horizontal', 'vertical'],
            [img_with_points, img_brightness, img_flip_horizontal, img_flip_vertical]
        ):
            buf = io.BytesIO()
            processed_img.save(buf, format='PNG')
            buf.seek(0)
            images_data[key] = base64.b64encode(buf.getvalue()).decode('ascii')

    except Exception as e:
        return f"Error al procesar la imagen: {str(e)}"

    # Pasar las imágenes generadas al template
    return render_template('resultado.html', images_data=images_data)



if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port= 5000)
