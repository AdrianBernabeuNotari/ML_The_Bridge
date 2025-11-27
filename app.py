# -*- coding: utf-8 -*-
import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import time
import io
import pandas as pd 
import datetime
import altair as alt # Necesario para el gráfico de donut

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(
    page_title="ML The Bridge",
    layout="wide", # ¡Esta línea asegura el ancho completo!
    initial_sidebar_state="expanded"
)

# --- INICIALIZACIÓN DEL MODELO ---
@st.cache_resource
def load_yolo_model():
    """Carga el modelo YOLOv8n una sola vez."""
    # Usamos YOLOv8n por su velocidad, crucial para el procesamiento en vivo.
    return YOLO("yolov8n.pt")

# Mapa de Clases de COCO (IDs y nombres relevantes para tu DEMO)
CLASES_INTERES = {
    # Clases de Tráfico y Personas (Iniciales)
    0: 'person',
    2: 'car',
    3: 'motorcycle',
    5: 'bus',
    7: 'truck',
    9: 'traffic light',
    
    # Clases de Demostración Solicitadas
    11: 'stop sign',
    27: 'backpack',
    39: 'bottle',
    46: 'banana',
    56: 'chair',
    63: 'laptop',
    64: 'mouse',
    67: 'cell phone',
    73: 'book'
}

# Mapeo de colores para el gráfico de barras (BGR)
COLOR_MAP_BGR = {
    'person': (0, 255, 255),  # Amarillo
    'car': (0, 0, 255),       # Rojo
    'motorcycle': (255, 0, 0), # Azul
    'bus': (0, 165, 255),     # Naranja
    'truck': (128, 0, 128),   # Púrpura
    'traffic light': (0, 255, 0), # Verde
    'stop sign': (0, 0, 128), # Granate
    'backpack': (203, 192, 255),# Rosa claro
    'bottle': (255, 255, 0),  # Cian
    'banana': (0, 200, 200),  # Ocre
    'chair': (10, 10, 10),    # Gris oscuro
    'laptop': (255, 192, 203), # Lavanda
    'mouse': (255, 255, 255), # Blanco
    'cell phone': (0, 128, 0), # Verde oscuro
    'book': (255, 0, 255)     # Magenta
}

# --- FUNCIONES DE PROCESAMIENTO ---

def get_class_metrics(model, results, classes_to_track):
    """Calcula el conteo de objetos detectados."""
    current_count = {name: 0 for name in classes_to_track.values()}
    
    for det in results[0].boxes:
        class_id = int(det.cls.cpu().numpy()[0])
        
        # Buscamos el nombre de la clase usando el ID del modelo
        try:
            nombre_clase = model.names[class_id]
        except IndexError:
            continue
        
        if nombre_clase in classes_to_track.values():
            current_count[nombre_clase] += 1
            
    return current_count

def display_counters(current_count):
    """Muestra los contadores en Streamlit usando st.metric."""
    
    # CORRECCIÓN: Si no hay conteos, no dibujamos las columnas ni el título.
    if not current_count:
        return
        
    st.markdown("### Conteo Detectado")
    
    num_cols = min(len(current_count), 6) 
    cols = st.columns(num_cols)
    
    for i, (nombre, count) in enumerate(current_count.items()):
        # Mapeo de Emojis extendido
        icon = "🚗" if nombre == 'car' else \
               "🚶‍♂️" if nombre == 'person' else \
               "🚦" if nombre == 'traffic light' else \
               "🏍️" if nombre == 'motorcycle' else \
               "🚌" if nombre == 'bus' else \
               "🚛" if nombre == 'truck' else \
               "🛑" if nombre == 'stop sign' else \
               "🎒" if nombre == 'backpack' else \
               "🍾" if nombre == 'bottle' else \
               "🍌" if nombre == 'banana' else \
               "🪑" if nombre == 'chair' else \
               "💻" if nombre == 'laptop' else \
               "🖱️" if nombre == 'mouse' else \
               "📱" if nombre == 'cell phone' else \
               "📚" if nombre == 'book' else "❓" 
               
        cols[i % num_cols].metric(label=f"{icon} {nombre.capitalize()}", value=count)


def process_image_mode(model, classes_to_track):
    """Modo para subir y procesar una imagen estática."""
    uploaded_file = st.file_uploader("Sube una imagen para detección", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Img a array para el modelo
        image = Image.open(uploaded_file).convert("RGB")
        image_array = np.array(image)
        
        st.image(image_array, caption="Imagen Original", use_container_width=True)

        with st.spinner("Analizando imagen..."):
            
            # 1. Inferencia
            # Umbral de confianza ajustado a 0.30
            results = model(image_array, verbose=False, conf=0.30, iou=0.5, classes=list(classes_to_track.keys()))
            
            # 2. Dibujar Bounding Boxes y Plotear Resultados
            annotated_frame = results[0].plot()
            
            # 3. Mostrar el Frame Anotado
            st.image(
                cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), 
                caption="Detecciones", 
                use_container_width=True
            )

            # 4. Mostrar Contadores
            current_count = get_class_metrics(model, results, classes_to_track)
            display_counters(current_count)

            # 5. Descargar la imagen
            img_to_download = Image.fromarray(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
            buffer = io.BytesIO()
            img_to_download.save(buffer, format="JPEG")
            
            st.download_button(
                label="Descargar imagen procesada",
                data=buffer.getvalue(),
                file_name="yolo_detection.jpg",
                mime="image/jpeg"
            )


def process_video_mode(model, classes_to_track):
    """Modo para subir y procesar un video (archivo)."""
    uploaded_file = st.file_uploader("Sube un archivo de video (MP4/MOV)", type=["mp4", "mov"])

    st.info("⚠️ Procesar videos largos puede tardar o agotar el tiempo de espera de Streamlit.")

    if uploaded_file is not None:
        # Guardar temporalmente el archivo para que YOLO lo pueda leer
        temp_file = f"temp_video_{time.time()}_{uploaded_file.name}"
        with open(temp_file, "wb") as f:
            f.write(uploaded_file.read())
        
        st.video(temp_file, format='video/mp4', start_time=0)

        with st.spinner("Analizando video y generando detecciones (esto puede tardar)..."):
            
            # 1. Inferencia: YOLO procesa el video y guarda el resultado automáticamente
            results_path = model.predict(
                source=temp_file, 
                save=True,              # Guarda el video de salida
                verbose=False, 
                # Umbral de confianza ajustado a 0.30
                conf=0.30, 
                iou=0.5, 
                classes=list(classes_to_track.keys())
            )
            
            # 2. Encontrar la ruta del video resultante (el último 'predict' folder)
            # YOLO guarda el resultado en runs/detect/predictX/temp_video_...mp4
            last_predict_dir = sorted(os.listdir('runs/detect'))[-1]
            
            # CORRECCIÓN: Usamos el nombre del archivo temporal (temp_file) para la salida, 
            # ya que YOLO nombra el resultado según el archivo que procesa.
            output_video_path = os.path.join('runs/detect', last_predict_dir, os.path.basename(temp_file))
            
            st.success("✅ Detección completada.")

            # 3. Mostrar el video resultante si existe
            if os.path.exists(output_video_path):
                st.video(output_video_path, caption="Video Procesado", format="video/mp4")
                
                # 4. Ofrecer descarga
                with open(output_video_path, "rb") as video_file:
                    st.download_button(
                        label="Descargar Video Procesado",
                        data=video_file,
                        file_name=f"yolo_detection_{uploaded_file.name}",
                        mime="video/mp4"
                    )
            else:
                st.error("Error al encontrar el video de salida.")

        # Limpiar el archivo temporal
        os.remove(temp_file)

def live_detection_mode(model, classes_to_track):
    """Modo para detección en vivo con la webcam (código actual)."""
    
    # Inicializa la captura de video (0 = cámara web predeterminada)
    cap = cv2.VideoCapture(0)
    
    # --- CORRECCIONES DE ESTABILIDAD DEL STREAM ---
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    # --- Título de la Sección de Video ---
    st.markdown("### Detección en Vivo")
    
    # 1. DEFINIR LA ESTRUCTURA DE COLUMNAS (VIDEO Y ANÁLISIS)
    col_video, col_analysis = st.columns([3, 2]) # 60% Video, 40% Análisis

    with col_video:
        video_placeholder = st.empty()
    
    with col_analysis:
        # Aquí irá el Conteo (metrics) y debajo el Gráfico de Línea
        counter_placeholder = st.empty() 
        st.markdown("### Evolución de Detecciones (Historial)")
        chart_placeholder = st.empty() 

    # Placeholder para los contadores (Debajo del video) -> Eliminado, ahora va en col_analysis

    # Mensaje de estado
    status_text = st.sidebar.empty()

    if not cap.isOpened():
        status_text.error("🚨 Error: No se pudo abrir la cámara. Asegúrate de que no está en uso.")
        return

    status_text.success("✅ Cámara lista. Presiona el botón para detener.")
    
    # Inicializar el DataFrame de historial si no existe
    if 'detection_history_df' not in st.session_state:
        st.session_state.detection_history_df = pd.DataFrame(columns=['Time'] + list(classes_to_track.values()))

    # Bucle principal de detección
    while cap.isOpened() and st.session_state.running:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Corrección: Voltear el frame horizontalmente (común para corregir cámaras)
        frame = cv2.flip(frame, 1)

        # Inferencia con YOLO 
        # Umbral de confianza ajustado a 0.30
        results = model(frame, verbose=False, conf=0.30, iou=0.5, classes=list(classes_to_track.keys()))

        # 1. Dibujar Bounding Boxes y Plotear Resultados
        annotated_frame = results[0].plot()

        # 2. Conteo de Objetos
        current_count = get_class_metrics(model, results, classes_to_track)
        
        # 3. Implementación del Conteo DENTRO del Video (Esquina superior derecha)
        
        # Corrección de Unicode: Usamos solo el nombre de la clase en mayúsculas y el conteo
        x_start_count = annotated_frame.shape[1] - 10 # Empezar cerca del borde derecho
        y_start_count = 30 # Empezar cerca del borde superior
        
        # Iterar sobre el conteo para dibujar
        for nombre, count in current_count.items():
            if count > 0:
                # Usamos el nombre de la clase y el conteo (ej: PERSON: 1)
                text_to_display = f"{nombre.upper()}: {count}"
                
                # Fondo negro para el texto (opcional, pero mejora la visibilidad)
                (text_width, text_height), baseline = cv2.getTextSize(
                    text_to_display, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2 # Fuente ligeramente más pequeña
                )
                
                # Dibujar un rectángulo detrás del texto
                cv2.rectangle(
                    annotated_frame, 
                    (annotated_frame.shape[1] - text_width - 20, y_start_count - text_height - 10), # Posición ajustada
                    (annotated_frame.shape[1] - 10, y_start_count + 10), 
                    (0, 0, 0), # Color negro
                    cv2.FILLED
                )
                
                # Dibujar el texto blanco sobre el fondo
                cv2.putText(
                    annotated_frame, 
                    text_to_display, 
                    (annotated_frame.shape[1] - text_width - 15, y_start_count), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, 
                    (255, 255, 255), # Color blanco
                    2
                )
                
                y_start_count += 30 # Desplazar la siguiente línea hacia abajo

        # --- IMPLEMENTACIÓN DEL GRÁFICO DE BARRAS EN LA ESQUINA INFERIOR IZQUIERDA ---
        
        frame_h, frame_w, _ = annotated_frame.shape
        margin = 15
        bar_height = 15
        text_offset = 5
        graph_width = 150
        
        # Filtrar solo las clases que tienen conteo > 0 y ordenar por conteo
        active_counts = {k: v for k, v in current_count.items() if v > 0}
        
        # Tomar las 5 clases más detectadas para mantener el gráfico limpio y visible
        top_classes = sorted(active_counts.items(), key=lambda item: item[1], reverse=True)[:5]
        
        if top_classes:
            max_count = max(c for _, c in top_classes) or 1
            
            # Área base para el gráfico (ligeramente más grande para que quepan todas las barras)
            graph_total_height = margin + (len(top_classes) * (bar_height + margin))
            graph_area_top = frame_h - graph_total_height - margin
            
            # 2. Dibujar las barras y texto
            bar_y_offset = graph_area_top + margin # Empezar desde la parte superior del área del gráfico
            
            for nombre, count in top_classes:
                color = COLOR_MAP_BGR.get(nombre, (150, 150, 150)) # Color por defecto: Gris
                normalized_width = int((count / max_count) * graph_width)
                
                # Crear una capa temporal para la barra con transparencia
                bar_overlay = annotated_frame.copy()
                
                # Dibujar barra
                cv2.rectangle(
                    bar_overlay, 
                    (margin + 5, bar_y_offset), 
                    (margin + 5 + normalized_width, bar_y_offset + bar_height), 
                    color, 
                    cv2.FILLED
                )
                
                # Combinar la barra con **mayor opacidad** (0.7 en vez de 0.4)
                cv2.addWeighted(bar_overlay, 0.7, annotated_frame, 0.3, 0, annotated_frame)
                
                # Dibujar texto (Nombre de clase y Conteo)
                cv2.putText(
                    annotated_frame, 
                    f"{nombre.capitalize()} ({count})", 
                    (margin + 5, bar_y_offset - text_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.4, 
                    (255, 255, 255), # Texto Blanco
                    1
                )
                
                bar_y_offset += bar_height + margin # Siguiente barra
        
        # 4. Mostrar el Frame Anotado en Streamlit (TARGET: col_video)
        with col_video:
            video_placeholder.image(
                cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), 
                channels="RGB", 
                caption="Detección en Vivo", 
                output_format="PNG",
                use_container_width=True
            )
        
        # 5. Los Contadores externos (st.metric) (TARGET: col_analysis)
        with col_analysis:
            with counter_placeholder.container():
                display_counters(current_count)

        # 6. Actualizar el Historial para el Gráfico de Evolución
        
        # Crear la nueva fila de datos
        new_row = {'Time': datetime.datetime.now()}
        for name in st.session_state.detection_history_df.columns[1:]:
             new_row[name] = current_count.get(name, 0)
        
        # Añadir la nueva fila al DataFrame
        new_df = pd.DataFrame([new_row])
        st.session_state.detection_history_df = pd.concat([st.session_state.detection_history_df, new_df], ignore_index=True)

        # Limitar el historial a los últimos 50 puntos para no sobrecargar el gráfico
        if len(st.session_state.detection_history_df) > 50:
             st.session_state.detection_history_df = st.session_state.detection_history_df.tail(50)

        # 7. Dibujar el Gráfico de Evolución (Línea) (TARGET: col_analysis)
        with col_analysis:
            with chart_placeholder:
                st.line_chart(st.session_state.detection_history_df.set_index('Time'))
            
        time.sleep(0.01)

    # Liberar recursos cuando el bucle termina
    cap.release()
    st.session_state.running = False 
    
    # Limpiar el gráfico dinámico después de parar
    chart_placeholder.empty()


# --- INTERFAZ DE STREAMLIT ---

def main():
    st.title("Proyecto ML: detección de objetos con YOLOv8n")

    # Cargar el modelo
    try:
        model = load_yolo_model()
    except Exception as e:
        st.error(f"Error al cargar el modelo YOLOv8n: {e}")
        return

    # Inicializar historial de detección si no existe
    if 'detection_history_df' not in st.session_state:
        st.session_state.detection_history_df = pd.DataFrame(columns=['Time'] + list(CLASES_INTERES.values()))


    # --- NAVEGACIÓN EN LA BARRA LATERAL ---
    with st.sidebar:
        st.header("Selección de Modo")
        detection_mode = st.radio(
            "Selecciona la fuente de datos:",
            ["Webcam (Live)", "Imagen (Archivo)", "Video (Archivo)"],
            key='detection_mode'
        )
        st.markdown("---")
        st.header("Configuración")
        st.write("Modelo: YOLOv8n")
        
        # Filtro de clases opcional
        st.subheader("Clases Activas")
        clases_activas = {}
        
        # Ordenamos las claves para que las clases aparezcan de forma lógica en la sidebar
        sorted_keys = sorted(CLASES_INTERES.keys())

        for id in sorted_keys:
            name = CLASES_INTERES[id]
            # Ponemos por defecto activo 'person' y 'car' para que el demo sea útil desde el inicio
            default_state = name in ['person', 'car', 'traffic light']

            if st.checkbox(f"Detectar {name.capitalize()}", value=default_state, key=f"check_{id}"):
                clases_activas[id] = name
        
    # --- CONTROL DE EJECUCIÓN ---
    if 'running' not in st.session_state:
        st.session_state.running = False

    # Botones Iniciar/Detener solo para modos de streaming (Webcam)
    if detection_mode == "Webcam (Live)":
        
        # Usamos dos columnas pequeñas para colocar los botones horizontalmente
        btn_col1, btn_col2 = st.columns([1, 1])

        with btn_col1:
            if st.button("🔴 Iniciar Detección"):
                st.session_state.running = True
                # Limpiar el DataFrame de historial al inicio
                st.session_state.detection_history_df = pd.DataFrame(columns=['Time'] + list(clases_activas.values()))
        
        with btn_col2:
            # El botón de detener solo aparece si ya está corriendo
            if st.session_state.running:
                if st.button("⏹️ Detener Detección"):
                    st.session_state.running = False
                    # Recarga la página para liberar la cámara
                    st.rerun() 
        
        # Ejecución del modo Webcam
        if st.session_state.running:
            live_detection_mode(model, clases_activas)
        else:
            # Limpiar el historial si no está corriendo para que no aparezca el gráfico vacío
            st.session_state.detection_history_df = pd.DataFrame(columns=['Time'] + list(CLASES_INTERES.values()))
            
            st.markdown(
                """
                Pulsa **Iniciar Detección** para activar la cámara en modo **Webcam (Live)**.
                """
            )

    # Ejecución de los modos de archivo
    elif detection_mode == "Imagen (Archivo)":
        # Aseguramos que el estado de streaming esté desactivado
        st.session_state.running = False 
        process_image_mode(model, clases_activas)
        
    elif detection_mode == "Video (Archivo)":
        # Aseguramos que el estado de streaming esté desactivado
        st.session_state.running = False 
        process_video_mode(model, clases_activas)


if __name__ == "__main__":
    # Necesario para manejar archivos temporales de video
    import os
    if not os.path.exists('runs/detect'):
        os.makedirs('runs/detect')
        
    main()

# cd Bootcamp\ML_The_Bridge
# streamlit run app.py