🚗 Proyecto Machine Learning: Detección de Objetos con YOLOv8n  
  
📋 Descripción General del Proyecto  
  
Este repositorio contiene el código y los recursos utilizados para el proyecto final de Machine Learning, enfocado en la detección y monitorización de objetos en tiempo real (Webcam, imágenes y videos) utilizando el modelo pre-entrenado YOLOv8n (Nano) de Ultralytics.  
  
El objetivo principal es demostrar la capacidad de un modelo ligero para realizar inferencias rápidas y proporcionar métricas visuales dinámicas (conteo y gráficos de evolución) a través de una aplicación web interactiva desarrollada con Streamlit.  
  
🚀 Estructura del Repositorio  
  
La estructura del repositorio está organizada para separar el código de la aplicación, los modelos, los datos de prueba y el historial de desarrollo..  
├── app.py                      # Aplicación Streamlit final (principal)  
├── yolov8n.pt                  # Modelo YOLOv8n pre-entrenado  
├── notebook/                   # Notebooks de Jupyter para pruebas iniciales y validación  
├── imagenes_prueba/            # Imágenes estáticas para probar la detección  
├── runs/                       # Carpeta de salida de YOLO (detecciones de video)  
├── aproximaciones/             # Historial de notebooks y pruebas descartadas (ML/DS)  
├── app_copia.py                # Backup de una versión funcional de app.py  
├── demo/                       # Aplicación Streamlit de la primera demo (histórica)  
├── modelos/                    # (Carpeta Descartada) Iba a contener modelos entrenados  
├── LICENCE                     # Licencia del proyecto (MIT)  
└── README.md                   # Este archivo  
  
🛠️ Requisitos e Instalación  
  
Para ejecutar la aplicación Streamlit y reproducir la detección, necesitas tener Python instalado (se recomienda Python 3.9+).  
- Clonar el Repositorio:
> git clone [https://docs.github.com/es/repositories/creating-and-managing-repositories/quickstart-for-repositories](https://docs.github.com/es/repositories/creating-and-managing-repositories/quickstart-for-repositories)  
> cd [nombre-del-repositorio]  
- Instalar Dependencias:  
> pip install -r requirements.txt  
(Asegúrate de crear un archivo requirements.txt con las siguientes librerías: streamlit, ultralytics, opencv-python, pandas, numpy, Pillow).  
  
▶️ Uso de la Aplicación (Streamlit)  
  
La aplicación principal se ejecuta a través de app.py.  
  
Modo Webcam (Live)  
La aplicación se inicia directamente en modo webcam, mostrando el stream de tu cámara junto a gráficos y contadores en tiempo real.  
- Asegúrate de que no haya otras aplicaciones utilizando la cámara.  
- Ejecuta el siguiente comando en tu terminal:  
> streamlit run app.py  
- Una vez cargada en el navegador, selecciona las Clases Activas en la barra lateral y pulsa 🔴 Iniciar Detección.  
  
Modos Imagen y Video  
La barra lateral te permite cambiar el modo de detección para:  
- Imagen (Archivo): Sube un archivo JPG o PNG para una detección estática.  
- Video (Archivo): Sube un archivo MP4 o MOV para procesar todo el video y guardar el resultado con las cajas delimitadoras.  
  
📊 Características de la Interfaz  
  
La aplicación app.py utiliza Streamlit para ofrecer las siguientes funcionalidades en tiempo real:  
- Detección YOLOv8n: Realiza inferencia en tiempo real o en archivos estáticos con un umbral de confianza ajustado (conf=0.30) para garantizar la detección de objetos pequeños.  
- Visualización In-Video: Muestra el conteo de las clases detectadas y un gráfico de barras semi-transparente en la esquina del video para una vista rápida de la distribución.  
- Conteo de Métricas: Usa widgets st.metric (debajo del video) para un conteo claro de los objetos activos.  
- Gráfico de Evolución: Muestra un gráfico de líneas (Historial) que rastrea la evolución del conteo de objetos a lo largo del tiempo de la sesión, ideal para análisis de tendencias.  
  
📜 Licencia 
  
Este proyecto está bajo la Licencia MIT.  

    
Desarrollado para el Bootcamp de Data Science.
