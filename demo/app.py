import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
from io import BytesIO

import warnings
warnings.filterwarnings("ignore")

# Título
st.title("Proyecto ML: demo coches y señales")

# Texto
st.write("Detección de objetos con YOLOv8m en imágenes")

# Cargar modelos (= Colab)
ruta = r"C:\Users\adria\Bootcamp\ML_The_Bridge"
modelo_coches = YOLO("yolov8m.pt")
modelo_señales = YOLO(f"{ruta}/modelos/best_signs.pt")

# Seleccionar modelo
modelo_seleccionado = st.selectbox(
    "Modelo a usar",
    ["coches", "señales", "ambos"]
)


# Subir una imagen
imagen_demo = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])

# Leer imagen
if imagen_demo is not None:

    # Img a array porque patata
    imagen = Image.open(imagen_demo).convert("RGB")
    imagen_array = np.array(imagen)

    st.image(imagen_array, caption="Imagen demo", use_column_width=True)

    # Circulito de carga que queda chido
    with st.spinner("Analizando imagen..."):

        detec_total = []

        # Modelo coches
        if modelo_seleccionado in ["coches", "ambos"]:
            detec1 = modelo_coches(imagen_array)
            for box in detec1[0].boxes:
                nombre = detec1[0].names[box.cls[0].item()]
                if nombre == "stop sign":
                     continue
                detec_total.append(("coche", box))


        # Modelo señales
        if modelo_seleccionado in ["señales", "ambos"]:
             detec2 = modelo_señales(imagen_array)
             for box in detec2[0].boxes:
                  detec_total.append(("sign", box))

        # Dibujar bounding boxes
        imagen_mod = imagen_array.copy()

        for tipo, box in detec_total:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            color = (255, 0, 0) if tipo == "coche" else (0, 0, 255)

            cv2.rectangle(imagen_mod, (x1, y1), (x2, y2), color, 2)
            cv2.putText(imagen_mod, tipo, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        st.image(imagen_mod, caption="Detecciones", use_column_width=True)

    # Descargar la imagen
    # 1. convertir array a imagen
    imagen_con = Image.fromarray(imagen_mod)
    buffer = BytesIO()
    imagen_con.save(buffer, format="JPEG")
    imagen_bytes = buffer.getvalue()

    # 2. descargar
    st.download_button(
        label="Descargar imagen procesada",
        data=imagen_bytes,
        file_name="resultado.jpg",
        mime="image/jpeg"
    )

# Para tirar:
# cd Bootcamp/ML_The_Bridge/demo
# streamlit run app.py

'''
# -*- coding: utf-8 -*-
import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import time

# Título
st.title("Proyecto de Machine Learning")

# Texto
st.write("Detección de objetos con YOLOv8n en imágenes")

# Cargar modelos
ruta = r"C:\Users\adria\Bootcamp\ML_The_Bridge"
modelo = YOLO("yolov8.pt")

# Seleccionar método de entrada
modelo_seleccionado = st.selectbox(
    "Modelo a usar",
    ["imagen", "video", "webcam"]
)
'''