import os
import cv2
import numpy as np
import json
import psutil  # <--- IMPORTANTE: Nueva librería para métricas
import time
from datetime import datetime
from insightface.app import FaceAnalysis
import pickle


# --- CLASE PARA GESTIÓN DE MÉTRICAS ---
class SystemMonitor:
    def __init__(self, log_file="medidas.json"):
        self.log_file = log_file
        # Inicializamos el archivo si no existe
        if not os.path.exists(self.log_file):
            with open(self.log_file, "w") as f:
                json.dump([], f)

    def get_system_metrics(self):
        """Obtiene uso de CPU y RAM actuales"""
        # interval=None hace que la lectura de CPU sea no bloqueante
        cpu_usage = psutil.cpu_percent(interval=None)
        ram = psutil.virtual_memory()
        return cpu_usage, ram.percent, ram.used / (1024 * 1024)  # MB usados

    def log_detection(self, person_name, confidence, is_known):
        """Registra una detección y el estado del sistema en ese momento"""
        cpu, ram_percent, ram_mb = self.get_system_metrics()

        entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
            "tipo_deteccion": "Conocido" if is_known else "Desconocido",
            "identidad_predicha": person_name if person_name else "N/A",
            "nivel_confianza": float(confidence),  # Precisión del modelo (0.0 a 1.0)
            "metricas_sistema": {
                "cpu_percent": cpu,
                "ram_percent": ram_percent,
                "ram_usada_mb": round(ram_mb, 2),
            },
        }

        self._save_to_json(entry)

    def _save_to_json(self, new_entry):
        """Guarda en el archivo JSON (Append eficiente)"""
        try:
            # Leemos datos existentes
            data = []
            if os.path.exists(self.log_file) and os.path.getsize(self.log_file) > 0:
                with open(self.log_file, "r") as f:
                    try:
                        data = json.load(f)
                    except json.JSONDecodeError:
                        data = []

            data.append(new_entry)

            # Escribimos de nuevo (En producción real usaríamos una base de datos o logs rotativos)
            with open(self.log_file, "w") as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            print(f"Error guardando métricas: {e}")


# --- SISTEMA PRINCIPAL ---
class FaceRecognitionSystem:
    def __init__(
        self,
        known_faces_dir="identified-face",
        unknown_faces_dir="not-identified",
        embeddings_file="face_embeddings.pkl",
    ):
        self.known_faces_dir = known_faces_dir
        self.unknown_faces_dir = unknown_faces_dir
        self.embeddings_file = embeddings_file

        # Inicializar Monitor de Métricas
        self.monitor = SystemMonitor("medidas.json")

        os.makedirs(known_faces_dir, exist_ok=True)
        os.makedirs(unknown_faces_dir, exist_ok=True)

        print("Cargando modelo InsightFace (buffalo_l)...")
        # Mantenemos buffalo_l por tu requerimiento de precisión
        self.app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        self.app.prepare(ctx_id=-1, det_size=(480, 480))  # Resolución balanceada

        self.known_embeddings = {}
        self.load_and_update_embeddings()

        self.similarity_threshold = 0.4

    def load_and_update_embeddings(self):
        if os.path.exists(self.embeddings_file):
            print(f"Cargando embeddings...")
            try:
                with open(self.embeddings_file, "rb") as f:
                    self.known_embeddings = pickle.load(f)
            except Exception:
                self.known_embeddings = {}
        self.register_new_faces()

    def register_new_faces(self):
        if not os.path.exists(self.known_faces_dir):
            return

        image_files = [
            f
            for f in os.listdir(self.known_faces_dir)
            if f.lower().endswith((".jpg", ".png"))
        ]
        changes = False

        for img_file in image_files:
            name = os.path.splitext(img_file)[0]
            if name in self.known_embeddings:
                continue

            img = cv2.imread(os.path.join(self.known_faces_dir, img_file))
            if img is None:
                continue

            faces = self.app.get(img)
            if len(faces) > 0:
                # Ordenar por tamaño para tomar la cara principal
                faces = sorted(
                    faces,
                    key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]),
                    reverse=True,
                )
                self.known_embeddings[name] = faces[0].normed_embedding
                print(f"Registrado: {name}")
                changes = True

        if changes:
            with open(self.embeddings_file, "wb") as f:
                pickle.dump(self.known_embeddings, f)

    def identify_face(self, face_embedding):
        if not self.known_embeddings:
            return None, 0.0

        best_match = None
        best_sim = -1

        for name, emb in self.known_embeddings.items():
            sim = np.dot(face_embedding, emb) / (
                np.linalg.norm(face_embedding) * np.linalg.norm(emb)
            )
            if sim > best_sim:
                best_sim = sim
                best_match = name

        if best_sim > self.similarity_threshold:
            return best_match, best_sim
        return None, best_sim

    def run_recognition(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error cámara")
            return

        print("\nIniciando sistema con REGISTRO DE MÉTRICAS...")

        # Variables para optimización (Frame Skipping)
        PROCESS_EVERY_N_FRAMES = 15  # Ajustable según temperatura de la RPi
        frame_count = 0
        last_results = []

        # Cooldown para no llenar el JSON de métricas con datos repetidos en milisegundos
        last_log_time = 0
        LOG_COOLDOWN = 1.0  # Registrar métricas máximo 1 vez por segundo

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            current_time = time.time()

            # --- LÓGICA DE DETECCIÓN (Solo cada N frames) ---
            if frame_count % PROCESS_EVERY_N_FRAMES == 0:
                faces = self.app.get(frame)
                last_results = []

                for face in faces:
                    name, conf = self.identify_face(face.normed_embedding)
                    box = face.bbox.astype(int)

                    is_known = name is not None
                    display_name = name if is_known else "Desconocido"

                    last_results.append((box, display_name, conf, is_known))

                    # --- REGISTRO DE MÉTRICAS ---
                    # Solo registramos si ha pasado el tiempo de cooldown para no colapsar el archivo
                    if current_time - last_log_time > LOG_COOLDOWN:
                        # Logueamos tanto conocidos como desconocidos para análisis de precisión
                        self.monitor.log_detection(name, conf, is_known)
                        last_log_time = current_time

            frame_count += 1

            # --- DIBUJADO ---
            for box, name, conf, is_known in last_results:
                color = (0, 255, 0) if is_known else (0, 0, 255)
                x1, y1, x2, y2 = box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame,
                    f"{name} {conf:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                )

            # Mostrar info de sistema en pantalla también (opcional)
            if frame_count % 30 == 0:  # Actualizar info en consola cada ~1 seg
                cpu, ram, _ = self.monitor.get_system_metrics()
                print(f"Estado Sistema -> CPU: {cpu}% | RAM: {ram}%")

            cv2.imshow("Monitor de Rendimiento AI", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    SystemMonitor()  # Crea el archivo si no existe
    sys = FaceRecognitionSystem()
    sys.run_recognition()
