import cv2
import numpy as np
import pickle
from insightface.app import FaceAnalysis
from pathlib import Path
import time
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import uvicorn
from threading import Thread
from queue import Queue
from datetime import datetime

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class CameraStream:
    def __init__(self, source=0):
        self.source = source
        self.status = False
        self.frame_queue = Queue(maxsize=2)
        self.capture = None

    def start(self):
        self.capture = cv2.VideoCapture(self.source)
        if not self.capture.isOpened():
            raise Exception(f"Could not connect to camera: {self.source}")
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        self.status = True
        self.thread = Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        return True

    def _capture_loop(self):
        while self.status:
            if not self.frame_queue.full():
                ret, frame = self.capture.read()
                if ret:
                    if self.frame_queue.full():
                        try:
                            self.frame_queue.get_nowait()
                        except Queue.Empty:
                            pass
                    self.frame_queue.put(frame)
            time.sleep(0.001)

    def read(self):
        if not self.status:
            return False, None
        try:
            frame = self.frame_queue.get(timeout=1.0)
            return True, frame
        except:
            return False, None

    def release(self):
        self.status = False
        if self.capture is not None:
            self.capture.release()

class FaceRecognitionSystem:
    def __init__(self, dataset_path="dataset", threshold=0.4, embeddings_file="face_embeddings.pkl"):
        self.threshold = threshold
        self.app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        self.embeddings_file = embeddings_file
        self.known_faces, self.known_labels = self.load_embeddings(dataset_path)
        self.last_detection_times = {}

    def load_embeddings(self, dataset_path):
        dataset_path = Path(dataset_path)
        if Path(self.embeddings_file).exists():
            with open(self.embeddings_file, "rb") as f:
                data = pickle.load(f)
                return data["embeddings"], data["labels"]
        
        known_faces = []
        known_labels = []
        for student_folder in dataset_path.iterdir():
            if student_folder.is_dir():
                student_name = student_folder.stem
                for img_path in student_folder.glob("*.jpg"):
                    img = cv2.imread(str(img_path))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    face_info = self.app.get(img)
                    if face_info:
                        embedding = face_info[0].embedding
                        known_faces.append(embedding)
                        known_labels.append(student_name)

        with open(self.embeddings_file, "wb") as f:
            pickle.dump({"embeddings": known_faces, "labels": known_labels}, f)

        return known_faces, known_labels

    def recognize_face(self, face_embedding):
        if not self.known_faces:
            return "Unknown", 0.0

        face_embedding = face_embedding / np.linalg.norm(face_embedding)
        similarities = [np.dot(face_embedding, known_face) / (np.linalg.norm(known_face) + 1e-6) for known_face in self.known_faces]
        max_similarity = max(similarities)
        best_match_index = similarities.index(max_similarity)

        if max_similarity > self.threshold:
            return self.known_labels[best_match_index], max_similarity
        return "Unknown", max_similarity

    def process_frame(self, frame):
        faces = self.app.get(frame)
        current_detected_faces = []
        for face in faces:
            x1, y1, x2, y2 = face.bbox.astype(int)
            label, confidence = self.recognize_face(face.embedding)
            
            current_time = datetime.now()
            if label != "Unknown":
                last_detected = self.last_detection_times.get(label, None)
                if last_detected is None or (current_time - last_detected).total_seconds() > 300:
                    detected_faces.append({"name": label, "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S")})
                    self.last_detection_times[label] = current_time
            
            color = (0, 255, 0) if label != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} ({confidence:.2f})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return frame, current_detected_faces

stream = CameraStream(0)
stream.start()
face_system = FaceRecognitionSystem()
detected_faces = []

def generate_frames():
    while True:
        ret, frame = stream.read()
        if not ret:
            break
        frame, _ = face_system.process_frame(frame)
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/detected_faces")
async def get_detected_faces():
    return {"faces": detected_faces}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)