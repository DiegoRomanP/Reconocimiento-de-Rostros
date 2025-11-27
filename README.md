# 👁️ Sistema de Control de Acceso por Reconocimiento Facial

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green)
![InsightFace](https://img.shields.io/badge/InsightFace-SOTA%20Model-orange)
![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20Raspberry%20Pi-lightgrey)

Un sistema de reconocimiento facial en tiempo real robusto y optimizado, diseñado para control de asistencia o seguridad. Utiliza **InsightFace (ArcFace)** para la generación de embeddings vectoriales de alta precisión y **OpenCV** para el procesamiento de video.

Este proyecto implementa lógica de persistencia de datos, optimización de memoria (caché de embeddings) y gestión inteligente de registros para evitar redundancia, siendo compatible con entornos de **Edge AI** como Raspberry Pi 4/5.

## 🚀 Características Principales

* **Detección y Reconocimiento en Tiempo Real:** Uso del modelo `buffalo_l` para alta precisión.
* **Vector Embeddings:** Conversión de rostros a vectores de 512 dimensiones para comparación matemática.
* **Similitud Coseno:** Algoritmo matemático para determinar la identidad con un umbral ajustable.
* **Gestión Inteligente de Registros (Cooldown):** Evita el "spam" de registros en la base de datos JSON si la persona permanece frente a la cámara.
* **Captura de Intrusos:** Detecta y guarda automáticamente fotografías de rostros desconocidos (con limitador de frecuencia para ahorrar almacenamiento).
* **Sistema Híbrido de Carga:** Carga rápida mediante `pickle` y escaneo automático de nuevas imágenes en la carpeta de registro.

## 🛠️ Tecnologías Utilizadas

* **Lenguaje:** Python 3.x
* **Visión Computacional:** OpenCV (`cv2`)
* **Deep Learning / Model:** InsightFace (ONNX Runtime)
* **Procesamiento Numérico:** NumPy
* **Persistencia:** JSON (Logs) y Pickle (Embeddings Cache)

## 📂 Estructura del Proyecto

```text
.
├── identified-face/       # 📸 Coloca aquí las fotos de personas conocidas (ej: juan_perez.jpg)
├── not-identified/        # ⚠️ Aquí se guardan automáticamente los desconocidos
├── face_embeddings.pkl    # 🧠 Archivo caché de vectores (se genera solo)
├── access_records.json    # 📝 Log de accesos en formato JSON
├── main.py                # 🐍 Script principal
├── requirements.txt       # 📦 Dependencias
└── README.md              # 📄 Documentación
