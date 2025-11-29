import os
import cv2
import numpy as np
import json
import psutil
import time
from datetime import datetime
from insightface.app import FaceAnalysis
import pickle
from dotenv import load_dotenv

load_dotenv()


# --- MONITOR DE SISTEMA ---
class SystemMonitor:
    def __init__(self, log_file="medidas.json"):
        self.log_file = log_file
        if not os.path.exists(self.log_file):
            with open(self.log_file, "w") as f:
                json.dump([], f)

    def get_system_metrics(self):
        cpu_usage = psutil.cpu_percent(interval=None)
        ram = psutil.virtual_memory()
        return cpu_usage, ram.percent, ram.used / (1024 * 1024)

    def log_detection(self, person_name, confidence, is_known):
        cpu, ram_percent, ram_mb = self.get_system_metrics()
        entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
            "tipo_deteccion": "Conocido" if is_known else "Desconocido",
            "identidad": person_name if person_name else "N/A",
            "confianza": float(confidence),
            "sistema": {"cpu": cpu, "ram": ram_percent},
        }
        self._save_to_json(entry)

    def _save_to_json(self, new_entry):
        try:
            data = []
            if os.path.exists(self.log_file) and os.path.getsize(self.log_file) > 0:
                with open(self.log_file, "r") as f:
                    try:
                        data = json.load(f)
                    except:
                        data = []
            data.append(new_entry)
            with open(self.log_file, "w") as f:
                json.dump(data, f, indent=2)
        except:
            pass


# --- SISTEMA DE RECONOCIMIENTO ---
class FaceRecognitionSystem:
    def __init__(
        self,
        known_faces_dir="identified-face",
        unknown_faces_dir="not-identified",  # <--- NUEVO: Directorio recuperado
        embeddings_file="face_embeddings.pkl",
    ):
        self.known_faces_dir = known_faces_dir
        self.unknown_faces_dir = unknown_faces_dir
        self.embeddings_file = embeddings_file
        self.monitor = SystemMonitor("medidas.json")

        os.makedirs(known_faces_dir, exist_ok=True)
        os.makedirs(unknown_faces_dir, exist_ok=True)  # Crear carpeta si no existe

        print("Cargando modelo InsightFace...")
        self.app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        self.app.prepare(ctx_id=-1, det_size=(480, 480))

        self.known_embeddings = {}
        self.load_and_update_embeddings()

        # AJUSTE: He bajado un poco el umbral por si la cámara tiene luz diferente
        # Si sigue fallando, bájalo a 0.35
        self.similarity_threshold = 0.20

        def load_and_update_embeddings(self):
        # 1. Intentar cargar embeddings previos
        if os.path.exists(self.embeddings_file):
            print(f"📂 Cargando base de datos desde {self.embeddings_file}...")
            try:
                with open(self.embeddings_file, "rb") as f:
                    self.known_embeddings = pickle.load(f)
                print(f"   ✅ Se cargaron {len(self.known_embeddings)} perfiles existentes.")
            except Exception as e:
                print(f"   ⚠️ Error leyendo archivo (posiblemente corrupto): {e}")
                print("   🔄 Se creará una nueva base de datos.")
                self.known_embeddings = {}
        else:
            print("🆕 Creando nueva base de datos de rostros...")
            self.known_embeddings = {}

        # 2. Escanear carpeta buscando fotos nuevas
        if not os.path.exists(self.known_faces_dir):
            print(f"❌ Error: La carpeta '{self.known_faces_dir}' no existe.")
            return

        print(f"🔎 Escaneando carpeta '{self.known_faces_dir}'...")
        image_files = os.listdir(self.known_faces_dir)
        
        if not image_files:
            print("   ⚠️ La carpeta está vacía. No hay rostros para aprender.")
        
        changed = False
        
        for f in image_files:
            if not f.lower().endswith((".jpg", ".png", ".jpeg")):
                continue
                
            name = os.path.splitext(f)[0]
            
            # Si ya lo tenemos registrado, saltamos
            if name in self.known_embeddings:
                continue

            print(f"   Procesando: {f}...", end=" ")
            img_path = os.path.join(self.known_faces_dir, f)
            img = cv2.imread(img_path)

            if img is None:
                print("❌ Error: No se pudo leer la imagen (archivo dañado).")
                continue

            faces = self.app.get(img)

            if len(faces) == 0:
                print("⚠️ NO SE DETECTÓ ROSTRO. Esta imagen será ignorada.")
                continue
            
            # Si hay más de un rostro, tomamos el más grande
            faces = sorted(
                faces,
                key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]),
                reverse=True,
            )
            
            # Guardamos el embedding
            self.known_embeddings[name] = faces[0].normed_embedding
            print("✅ OK! Registrado.")
            changed = True

        # 3. Guardar cambios en el archivo .pkl
        if changed:
            print("💾 Guardando cambios en face_embeddings.pkl...")
            try:
                with open(self.embeddings_file, "wb") as f:
                    pickle.dump(self.known_embeddings, f)
                print("✅ Base de datos actualizada correctamente.")
            except Exception as e:
                print(f"❌ Error guardando el archivo .pkl: {e}")
        elif len(self.known_embeddings) == 0:
             print("⚠️ Advertencia: No hay rostros registrados. El sistema no reconocerá a nadie.")
        else:
            print("👌 No se encontraron cambios nuevos.")
        def identify_face(self, embedding):
        if not self.known_embeddings:
            return None, 0.0

        best_match = None
        best_sim = -1.0

        for name, emb in self.known_embeddings.items():
            sim = np.dot(embedding, emb) / (
                np.linalg.norm(embedding) * np.linalg.norm(emb)
            )
            if sim > best_sim:
                best_sim = sim
                best_match = name

        # DEBUG: Imprimir similitud para entender por qué falla
        # Esto te ayudará a ver si necesitas bajar el similarity_threshold
        if best_sim > 0.1:
            print(f"Mejor coincidencia: {best_match} con similitud: {best_sim:.2f}")

        if best_sim > self.similarity_threshold:
            return best_match, best_sim
        else:
            return None, best_sim

    # --- NUEVA FUNCIÓN RECUPERADA ---
    def save_unknown_face(self, face_img):
        """Guarda la imagen del rostro desconocido"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"unknown_{timestamp}.jpg"
        filepath = os.path.join(self.unknown_faces_dir, filename)
        try:
            cv2.imwrite(filepath, face_img)
            print(f"📸 Foto guardada: {filename}")
        except Exception as e:
            print(f"Error guardando foto: {e}")

    def draw_dashboard(self, frame, detection_info, metrics):
        h, w, _ = frame.shape
        panel_width = 300
        panel = np.zeros((h, panel_width, 3), dtype=np.uint8)

        WHITE = (255, 255, 255)
        GREEN = (0, 255, 0)
        RED = (0, 0, 255)
        GRAY = (200, 200, 200)

        cv2.putText(panel, "SISTEMA", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2)
        cv2.putText(
            panel,
            f"CPU: {metrics[0]}%",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            GRAY,
            1,
        )
        cv2.putText(
            panel,
            f"RAM: {metrics[1]}%",
            (10, 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            GRAY,
            1,
        )
        cv2.line(panel, (10, 100), (panel_width - 10, 100), GRAY, 1)

        cv2.putText(
            panel, "DETECCION", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2
        )

        if not detection_info:
            cv2.putText(
                panel, "Esperando...", (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, GRAY, 1
            )
        else:
            box, name, conf, is_known = detection_info[0]
            status_text = "ACCESO PERMITIDO" if is_known else "DESCONOCIDO"
            status_color = GREEN if is_known else RED

            cv2.rectangle(panel, (5, 145), (panel_width - 5, 175), status_color, -1)
            cv2.putText(
                panel,
                status_text,
                (10, 165),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 0),
                2,
            )
            cv2.putText(
                panel, "Identidad:", (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1
            )
            display_name = name if is_known else "No Identificado"
            cv2.putText(
                panel, display_name, (10, 235), cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2
            )
            cv2.putText(
                panel,
                f"Confianza: {conf:.2f}",
                (10, 265),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                GRAY,
                1,
            )

        return np.hstack((frame, panel))

    def run_recognition(self):
        user = os.getenv("HIK_USER")
        password = os.getenv("HIK_PASS")
        ip = os.getenv("HIK_IP")
        channel = os.getenv("HIK_CHANNEL")

        if not all([user, password, ip, channel]):
            print("❌ Error: Faltan variables en el archivo .env")
            return

        RTSP_URL = f"rtsp://{user}:{password}@{ip}:554/Streaming/Channels/{channel}"
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"
        print(f"Conectando a cámara en: rtsp://{user}:*****@{ip}:554...")

        cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)

        if not cap.isOpened():
            print("❌ Error: No se pudo conectar a la cámara IP.")
            return
        else:
            print("✅ Conexión exitosa con Hikvision")

        PROCESS_EVERY_N_FRAMES = 5
        frame_count = 0
        last_results = []
        last_metrics = (0, 0, 0)

        # Variables de tiempo para evitar spam
        last_log_time = 0
        LOG_COOLDOWN = 1.0

        last_unknown_save_time = 0
        UNKNOWN_SAVE_COOLDOWN = 5.0  # Guardar foto de desconocido máximo cada 5 seg

        print("Sistema iniciado. Presiona 'q' para salir.")

        while True:
            # Truco para vaciar buffer si hay delay
            # for _ in range(2): cap.read()

            ret, frame = cap.read()
            if not ret:
                break

            current_time = time.time()

            if frame_count % PROCESS_EVERY_N_FRAMES == 0:
                last_metrics = self.monitor.get_system_metrics()
                faces = self.app.get(frame)
                last_results = []

                for face in faces:
                    name, conf = self.identify_face(face.normed_embedding)
                    box = face.bbox.astype(int)
                    is_known = name is not None

                    # Guardamos resultados para pintar
                    last_results.append((box, name, conf, is_known))

                    # 1. LOGGING (JSON)
                    if current_time - last_log_time > LOG_COOLDOWN:
                        self.monitor.log_detection(name, conf, is_known)
                        last_log_time = current_time

                    # 2. CAPTURA DE DESCONOCIDOS (IMAGEN)
                    # Si NO es conocido Y ha pasado el tiempo de cooldown
                    if not is_known and (
                        current_time - last_unknown_save_time > UNKNOWN_SAVE_COOLDOWN
                    ):
                        # Recortar la cara para guardar (con un margen pequeño si es posible)
                        x1, y1, x2, y2 = box
                        # Asegurar que las coordenadas estén dentro de la imagen
                        h_img, w_img, _ = frame.shape
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(w_img, x2), min(h_img, y2)

                        face_crop = frame[y1:y2, x1:x2]
                        if face_crop.size > 0:
                            self.save_unknown_face(face_crop)
                            last_unknown_save_time = current_time

            frame_count += 1

            for box, name, conf, is_known in last_results:
                x1, y1, x2, y2 = box
                color = (0, 255, 0) if is_known else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            final_ui = self.draw_dashboard(frame, last_results, last_metrics)
            cv2.imshow("Panel de Control - Acceso IA", final_ui)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    sys = FaceRecognitionSystem()
    sys.run_recognition()
