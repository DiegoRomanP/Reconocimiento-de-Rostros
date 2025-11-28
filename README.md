# 👁️ Sistema de Control de Acceso Facial con Telemetría (Edge AI)

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![InsightFace](https://img.shields.io/badge/InsightFace-SOTA%20Model-orange)
![Performance](https://img.shields.io/badge/Performance-RealTime%20Monitoring-brightgreen)
![Platform](https://img.shields.io/badge/Device-Raspberry%20Pi%204%2F5-lightgrey)

Un sistema de reconocimiento facial robusto diseñado para entornos de **Edge Computing**. Implementa un pipeline de visión computacional optimizado que separa la inferencia (IA) del renderizado (UI), permitiendo una ejecución fluida en hardware limitado como Raspberry Pi.

El proyecto incluye un **Dashboard de Rendimiento** integrado y un módulo de **Data Logging** para el análisis posterior de consumo de recursos (CPU/RAM) y métricas de precisión del modelo.

## 🚀 Características de Ingeniería

* **Arquitectura Detect & Track:** Implementación de *Frame Skipping* (procesamiento asíncrono simétrico) para mantener 30 FPS visuales mientras la inferencia corre a 3-5 FPS, evitando el *thermal throttling*.
* **Dashboard UI Integrado:** Interfaz gráfica profesional que separa el video de los metadatos. Muestra en tiempo real:
  * Estado de Salud del Hardware (CPU % / RAM %).
  * Identidad y Nivel de Confianza (Confidence Score).
  * Alertas visuales de acceso (Verde/Rojo).
* **Vector Embeddings (ArcFace):** Uso del modelo `buffalo_l` para generar vectores de 512 dimensiones, garantizando alta precisión (>99.5% en LFW) incluso en condiciones difíciles.
* **Telemetría y Data Science:** Registro automático en `medidas.json` de cada inferencia, vinculando la precisión del modelo con el estado del hardware en ese milisegundo exacto.

## 🛠️ Stack Tecnológico

* **Core:** Python 3.x
* **Visión:** OpenCV (`cv2`)
* **Model Serving:** InsightFace sobre ONNX Runtime (CPU Optimized)
* **Monitoring:** Psutil (Métricas de sistema)
* **Math:** NumPy (Cálculo de similitud coseno y manipulación de matrices)

## 📂 Estructura del Proyecto

```text
.
├── identified-face/       # 📸 Dataset: Imágenes de usuarios autorizados
├── not-identified/        # ⚠️ Dataset: Capturas automáticas de intrusos
├── face_embeddings.pkl    # 🧠 Cache de vectores (Serialización Pickle)
├── medidas.json           # 📊 Telemetría: Logs para análisis de Data Science
├── face_recognition.py    # 🐍 Código fuente principal
├── requirements.txt       # 📦 Dependencias del proyecto
└── README.md              # 📄 Documentación
