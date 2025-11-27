import os
import cv2
import numpy as np
import json
from datetime import datetime, timedelta
from insightface.app import FaceAnalysis
import pickle


class FaceRecognitionSystem:
    def __init__(
        self,
        known_faces_dir="identified-face",
        unknown_faces_dir="not-identified",
        embeddings_file="face_embeddings.pkl",
        records_file="access_records.json",
    ):
        """
        Sistema de reconocimiento facial usando InsightFace
        """
        self.known_faces_dir = known_faces_dir
        self.unknown_faces_dir = unknown_faces_dir
        self.embeddings_file = embeddings_file
        self.records_file = records_file

        # Crear directorios si no existen
        os.makedirs(known_faces_dir, exist_ok=True)
        os.makedirs(unknown_faces_dir, exist_ok=True)

        # Inicializar el modelo de InsightFace
        print("Cargando modelo InsightFace (puede tardar la primera vez)...")
        # Usamos CPUExecutionProvider para asegurar compatibilidad.
        # Si tienes GPU NVIDIA configurada en tu Arch Linux, cambia a ['CUDAExecutionProvider']
        self.app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        # la siguiente linea es para cuando hay graficos
        # self.app.prepare(ctx_id=0, det_size=(640, 640))
        self.app.prepare(ctx_id=-1, det_size=(160, 160))

        # Base de datos de embeddings
        self.known_embeddings = {}
        self.load_and_update_embeddings()

        # Umbral de similitud (ajustable)
        self.similarity_threshold = 0.4

    def load_and_update_embeddings(self):
        """Carga embeddings existentes y busca nuevas imágenes para registrar"""
        # 1. Cargar lo que ya existe en el pickle
        if os.path.exists(self.embeddings_file):
            print(f"Cargando embeddings desde {self.embeddings_file}...")
            try:
                with open(self.embeddings_file, "rb") as f:
                    self.known_embeddings = pickle.load(f)
                print(f"Se cargaron {len(self.known_embeddings)} personas de la caché.")
            except Exception as e:
                print(f"Error cargando pickle (se creará uno nuevo): {e}")
                self.known_embeddings = {}

        # 2. Verificar si hay imágenes nuevas en la carpeta que no estén en el pickle
        self.register_new_faces()

    def register_new_faces(self):
        """Registra rostros de la carpeta que aún no tienen embedding"""
        if not os.path.exists(self.known_faces_dir):
            return

        image_files = [
            f
            for f in os.listdir(self.known_faces_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        changes_made = False

        for img_file in image_files:
            person_name = os.path.splitext(img_file)[0]

            # Si ya tenemos el embedding de esta persona, saltamos para ahorrar tiempo
            if person_name in self.known_embeddings:
                continue

            print(f"Procesando nueva imagen: {img_file}...")
            img_path = os.path.join(self.known_faces_dir, img_file)
            img = cv2.imread(img_path)

            if img is None:
                print(f"Error: No se pudo leer {img_file}")
                continue

            # Detectar rostro
            faces = self.app.get(img)

            if len(faces) == 0:
                print(f"⚠️ No se detectó rostro en: {img_file}")
                continue

            # Usar el rostro más grande si hay varios (asumimos que es el sujeto principal)
            faces = sorted(
                faces,
                key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]),
                reverse=True,
            )

            self.known_embeddings[person_name] = faces[0].normed_embedding
            print(f"✅ Registrado nuevo: {person_name}")
            changes_made = True

        # Guardar embeddings actualizados solo si hubo cambios
        if changes_made:
            with open(self.embeddings_file, "wb") as f:
                pickle.dump(self.known_embeddings, f)
            print("Base de datos de embeddings actualizada.")
        else:
            print("No se encontraron imágenes nuevas para procesar.")

    def cosine_similarity(self, embedding1, embedding2):
        """Calcula similitud coseno entre dos embeddings"""
        return np.dot(embedding1, embedding2) / (
            np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
        )

    def identify_face(self, face_embedding):
        """Identifica un rostro comparando con la base de datos"""
        if not self.known_embeddings:
            return None, 0.0

        best_match = None
        best_similarity = -1

        for name, known_embedding in self.known_embeddings.items():
            similarity = self.cosine_similarity(face_embedding, known_embedding)
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = name

        if best_similarity > self.similarity_threshold:
            return best_match, best_similarity
        else:
            return None, best_similarity

    def save_access_record(self, person_name, confidence):
        """Guarda registro de acceso en archivo JSON"""
        record = {
            "nombre": person_name,
            "fecha_hora": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "confianza": float(confidence),  # Convertir numpy float a python float
        }

        try:
            if os.path.exists(self.records_file):
                with open(self.records_file, "r", encoding="utf-8") as f:
                    try:
                        records = json.load(f)
                    except json.JSONDecodeError:
                        records = []
            else:
                records = []

            records.append(record)

            with open(self.records_file, "w", encoding="utf-8") as f:
                json.dump(records, f, indent=2, ensure_ascii=False)

        except Exception as e:
            print(f"Error guardando registro: {e}")

    def save_unknown_face(self, face_img):
        """Guarda imagen de rostro no identificado"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"unknown_{timestamp}.jpg"
        filepath = os.path.join(self.unknown_faces_dir, filename)
        cv2.imwrite(filepath, face_img)
        print(f"📸 Rostro desconocido guardado: {filename}")

    def run_recognition(self):
        """Ejecuta el sistema de reconocimiento en tiempo real"""
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("Error: No se pudo abrir la cámara")
            return

        print("\n=== Sistema de Reconocimiento Iniciado ===")
        print("Presiona 'q' para salir")
        print("Presiona 'r' para recargar base de datos")

        # Control de tiempos para evitar spam de registros
        last_identified_time = {}
        last_unknown_save_time = datetime.min

        # Configuraciones de Cooldown (en segundos)
        KNOWN_COOLDOWN = 5  # Tiempo entre registros de acceso para la misma persona
        UNKNOWN_COOLDOWN = 5.0  # Tiempo entre fotos guardadas de desconocidos

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Detectar rostros
            faces = self.app.get(frame)
            current_time = datetime.now()

            for face in faces:
                bbox = face.bbox.astype(int)
                x1, y1, x2, y2 = bbox

                # Identificar rostro
                person_name, confidence = self.identify_face(face.normed_embedding)

                if person_name:
                    # --- CASO: CONOCIDO ---
                    color = (0, 255, 0)
                    label = f"{person_name} ({confidence:.2f})"

                    # Verificar cooldown para no llenar el JSON
                    last_time = last_identified_time.get(person_name, datetime.min)
                    if (current_time - last_time).total_seconds() > KNOWN_COOLDOWN:
                        self.save_access_record(person_name, confidence)
                        last_identified_time[person_name] = current_time
                        print(f"✓ Acceso registrado: {person_name}")

                else:
                    # --- CASO: DESCONOCIDO ---
                    color = (0, 0, 255)
                    label = f"Desconocido ({confidence:.2f})"

                    # Verificar cooldown para no llenar el disco duro de fotos
                    if (
                        current_time - last_unknown_save_time
                    ).total_seconds() > UNKNOWN_COOLDOWN:
                        face_crop = frame[
                            max(0, y1) : min(frame.shape[0], y2),
                            max(0, x1) : min(frame.shape[1], x2),
                        ]

                        if face_crop.size > 0:
                            self.save_unknown_face(face_crop)
                            last_unknown_save_time = current_time

                # Dibujar interfaz
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # Fondo para el texto para mejor legibilidad
                text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x1, y1 - 25), (x1 + text_size[0], y1), color, -1)
                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )

            cv2.imshow("Sistema de Acceso - LAB", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                print("\nRecargando base de datos...")
                self.load_and_update_embeddings()

        cap.release()
        cv2.destroyAllWindows()
        print("\nSistema finalizado correctamente.")


def main():
    system = FaceRecognitionSystem()

    # Validar que existan datos antes de iniciar cámara
    if not system.known_embeddings:
        print("\n⚠️  ADVERTENCIA: No hay rostros registrados.")
        print(f"Coloca imágenes .jpg en la carpeta '{system.known_faces_dir}'")
        # No retornamos inmediatamente, permitimos iniciar para probar la detección
        input("Presiona ENTER para iniciar la cámara (solo detectará desconocidos)...")

    system.run_recognition()


if __name__ == "__main__":
    main()
