import cv2
import numpy as np
from insightface.app import FaceAnalysis
import insightface
from pathlib import Path
import time
import torch
from threading import Thread
from queue import Queue
import urllib.request
import sys
from numpy.linalg import norm


class CameraStream:
    """Enhanced camera stream handler with support for multiple sources"""

    def __init__(self, source):
        self.source = source
        self.status = False
        self.frame_queue = Queue(maxsize=2)
        self.capture = None

    def start(self):
        """Initialize and start the camera stream"""
        print(f"Attempting to connect to camera source: {self.source}")

        # Try to interpret source as an integer (for webcams)
        try:
            if isinstance(self.source, str) and self.source.isdigit():
                self.source = int(self.source)
        except ValueError:
            pass

        # Initialize capture
        self.capture = cv2.VideoCapture(self.source)

        # Check if connection was successful
        if not self.capture.isOpened():
            print("Failed to open camera. Trying common URL formats...")

            # Try common IP camera URL formats
            url_formats = [
                f"http://{self.source}/video",
                f"http://{self.source}/video.mjpg",
                f"rtsp://{self.source}/live",
                f"rtsp://{self.source}/h264_stream",
                f"http://{self.source}:8080/video"
            ]

            for url in url_formats:
                print(f"Trying: {url}")
                self.capture = cv2.VideoCapture(url)
                if self.capture.isOpened():
                    print(f"Successfully connected using: {url}")
                    break

        if not self.capture.isOpened():
            raise Exception(
                f"Could not connect to camera source: {self.source}")

        # Set camera properties for better performance
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)

        # Start capture thread
        self.status = True
        self.thread = Thread(target=self._capture_loop, args=(), daemon=True)
        self.thread.start()

        print("Camera stream started successfully")
        return True

    def _capture_loop(self):
        """Continuous frame capture loop"""
        while self.status:
            if not self.frame_queue.full():
                ret, frame = self.capture.read()
                if ret:
                    # Clear queue if full
                    if self.frame_queue.full():
                        try:
                            self.frame_queue.get_nowait()
                        except Queue.Empty:
                            pass
                    self.frame_queue.put(frame)
                else:
                    print("Error reading frame. Attempting to reconnect...")
                    self._reconnect()
            time.sleep(0.001)  # Small sleep to prevent excessive CPU usage

    def _reconnect(self):
        """Attempt to reconnect to the camera"""
        print("Attempting to reconnect...")
        if self.capture is not None:
            self.capture.release()

        max_attempts = 5
        for attempt in range(max_attempts):
            print(f"Reconnection attempt {attempt + 1}/{max_attempts}")
            self.capture = cv2.VideoCapture(self.source)
            if self.capture.isOpened():
                print("Successfully reconnected")
                return True
            time.sleep(2)

        print("Failed to reconnect after multiple attempts")
        self.status = False
        return False

    def read(self):
        """Read a frame from the queue"""
        if not self.status:
            return False, None
        try:
            frame = self.frame_queue.get(timeout=1.0)
            return True, frame
        except:
            return False, None

    def release(self):
        """Release resources"""
        self.status = False
        if hasattr(self, 'thread'):
            self.thread.join()
        if self.capture is not None:
            self.capture.release()


class FaceRecognitionSystem:
    def __init__(self, known_dataset_path, threshold=0.4):
        """Initialize face recognition system."""
        self.threshold = threshold
        self.app = FaceAnalysis(name='buffalo_l', providers=[
                                'CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(640, 640))

        self.known_faces = []
        self.known_labels = []
        self.load_known_faces(known_dataset_path)

    def load_known_faces(self, dataset_path):
        """Load known faces from dataset, using folder names as labels."""
        dataset_path = Path(dataset_path)

        for student_folder in dataset_path.iterdir():
            if student_folder.is_dir():  # Ensure it's a directory
                student_name = student_folder.stem  # Folder name as label

                # Loop through images
                for img_path in student_folder.glob("*.jpg"):
                    img = cv2.imread(str(img_path))
                    # Convert to RGB
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                    face_info = self.app.get(img)
                    if face_info:
                        embedding = face_info[0].embedding
                        self.known_faces.append(embedding)
                        # Assign student name as label
                        self.known_labels.append(student_name)
                        # Debugging
                        print(
                            f"Loaded face: {student_name}, Embedding: {embedding[:5]}")

        print(f"Total known faces loaded: {len(self.known_faces)}")

    def recognize_face(self, face_embedding):
        """Recognize a face using cosine similarity."""
        if not self.known_faces:
            return "Unknown", 0.0

    # Normalize embeddings
        face_embedding = face_embedding / norm(face_embedding)

        similarities = [
            np.dot(face_embedding, known_face) / (norm(known_face) + 1e-6)
            for known_face in self.known_faces
        ]

        max_similarity = max(similarities)
        best_match_index = similarities.index(max_similarity)

        # Debugging
        print(
            f"Detected Face Similarity: {max_similarity}, Closest Match: {self.known_labels[best_match_index]}")

        # Adjust threshold (e.g., 0.6 - 0.8)
        if max_similarity > self.threshold:
            return self.known_labels[best_match_index], max_similarity
        else:
            return "Unknown", max_similarity

    def process_frame(self, frame):
        """Detect and recognize faces in a given frame."""
        faces = self.app.get(frame)
        for face in faces:
            x1, y1, x2, y2 = face.bbox.astype(int)
            label, confidence = self.recognize_face(face.embedding)

            color = (0, 255, 0) if label != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} ({confidence:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return frame

    def process_camera(self, camera_source):
        """Process camera feed with improved error handling."""
        print("\nInitializing camera feed...")
        stream = CameraStream(camera_source)

        try:
            if not stream.start():
                print("Failed to start camera stream")
                return

            print("\nStarting face recognition...")
            print("Press 'q' to quit")
            print("Press 's' to save a screenshot")

            frame_count = 0
            start_time = time.time()

            while True:
                ret, frame = stream.read()
                if not ret:
                    print("No frame received. Checking connection...")
                    time.sleep(1)
                    continue

                # Process frame
                processed_frame = self.process_frame(frame)

                # Calculate FPS
                frame_count += 1
                elapsed_time = time.time() - start_time
                fps = frame_count / elapsed_time

                # Add FPS to frame
                cv2.putText(processed_frame, f"FPS: {fps:.1f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Display frame
                cv2.imshow('Face Recognition', processed_frame)

                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("Quitting...")
                    break
                elif key == ord('s'):
                    # Save screenshot
                    screenshot_path = f"screenshot_{int(time.time())}.jpg"
                    cv2.imwrite(screenshot_path, processed_frame)
                    print(f"Screenshot saved: {screenshot_path}")

        except Exception as e:
            print(f"Error occurred: {str(e)}")
        finally:
            stream.release()
            cv2.destroyAllWindows()


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Camera Face Recognition System')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Path to dataset directory')
    parser.add_argument('--source', type=str, required=True,
                        help='Camera source (IP address, URL, or device number)')

    args = parser.parse_args()

    # Initialize face recognition system
    face_system = FaceRecognitionSystem(args.dataset)

    # Start processing camera feed
    face_system.process_camera(args.source)


if __name__ == "__main__":
    main()
