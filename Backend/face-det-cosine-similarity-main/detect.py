import cv2
import numpy as np
from insightface.app import FaceAnalysis
import insightface
from pathlib import Path
import time
import torch


class FaceRecognitionSystem:
    def __init__(self, known_dataset_path, threshold=0.4):
        # Check for CUDA availability
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Initialize InsightFace with GPU support
        self.face_analyzer = FaceAnalysis(
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.face_analyzer.prepare(ctx_id=0, det_size=(640, 640))

        # Recognition model for feature extraction with GPU support
        model_name = 'buffalo_l'
        self.recognition_model = insightface.model_zoo.get_model(model_name,
                                                                 providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.recognition_model.prepare(ctx_id=0)

        self.known_embeddings = {}
        self.threshold = threshold
        self.load_known_faces(known_dataset_path)

        # Batch size for processing
        self.batch_size = 4 if torch.cuda.is_available() else 1

    def load_known_faces(self, dataset_path):
        """Load and create embeddings for known faces."""
        dataset_path = Path(dataset_path)

        # Batch processing for known faces
        batch_images = []
        batch_names = []

        for student_dir in dataset_path.iterdir():
            if student_dir.is_dir():
                student_name = student_dir.name
                image_path = next(student_dir.glob(
                    '*.jpg'))  # Get first jpg image

                # Read image
                img = cv2.imread(str(image_path))
                if img is None:
                    print(f"Warning: Could not read image at {image_path}")
                    continue

                # Process single image immediately
                faces = self.face_analyzer.get(img)
                if faces:
                    face = faces[0]
                    # Align face
                    aligned_face = face.embedding
                    if aligned_face is not None:
                        self.known_embeddings[student_name] = aligned_face
                    else:
                        print(
                            f"Warning: Could not get embedding for {student_name}")
                else:
                    print(f"Warning: No face detected in {image_path}")

    def find_matching_student(self, face_embedding):
        """Find matching student using cosine similarity with GPU acceleration."""
        if torch.cuda.is_available():
            face_embedding = torch.tensor(face_embedding, device=self.device)

        max_similarity = -1
        matched_student = None

        # Convert embeddings to tensor for batch processing
        if torch.cuda.is_available():
            known_embeddings_tensor = torch.stack([torch.tensor(emb, device=self.device)
                                                   for emb in self.known_embeddings.values()])
            similarities = torch.nn.functional.cosine_similarity(
                face_embedding.unsqueeze(0),
                known_embeddings_tensor
            )

            max_similarity, max_idx = torch.max(similarities, dim=0)
            max_similarity = max_similarity.item()

            if max_similarity > self.threshold:
                matched_student = list(self.known_embeddings.keys())[max_idx]
        else:
            # Fall back to CPU processing
            for student_name, known_embedding in self.known_embeddings.items():
                similarity = self.cosine_similarity(
                    face_embedding, known_embedding)
                if similarity > max_similarity and similarity > self.threshold:
                    max_similarity = similarity
                    matched_student = student_name

        return matched_student, max_similarity

    @staticmethod
    def cosine_similarity(embedding1, embedding2):
        """Calculate cosine similarity between two embeddings."""
        if torch.is_tensor(embedding1):
            return torch.nn.functional.cosine_similarity(
                embedding1.unsqueeze(0),
                embedding2.unsqueeze(0)
            ).item()
        return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))

    def process_video(self, video_path, output_path=None, use_cuda=True, display=False):
        """Process video and detect/recognize faces with GPU acceleration."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception(f"Error: Could not open video file {video_path}")

        # Set up video writer if output path is provided
        if output_path:
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))

            # Ensure the output directory exists
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)

            # Try different codecs in order of preference
            codecs = [
                ('mp4v', '.mp4'),
                ('avc1', '.mp4'),
                ('XVID', '.avi'),
                ('MJPG', '.avi'),
                ('DIV3', '.avi'),
                ('X264', '.mp4'),
            ]

            out = None
            for codec, ext in codecs:
                try:
                    output_file = str(output_dir / f"output{ext}")
                    fourcc = cv2.VideoWriter_fourcc(*codec)
                    out = cv2.VideoWriter(
                        output_file, fourcc, fps, (frame_width, frame_height))
                    if out.isOpened():
                        print(
                            f"Successfully initialized video writer with codec {codec}")
                        break
                except Exception as e:
                    print(f"Failed to initialize codec {codec}: {str(e)}")
                    if out is not None:
                        out.release()

            if out is None or not out.isOpened():
                raise Exception(
                    "Failed to create output video file with any codec")

        try:
            # For FPS calculation
            prev_time = time.time()
            fps_counter = 0
            fps_display = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Detect faces in the frame
                faces = self.face_analyzer.get(frame)

                for face in faces:
                    # Get embedding directly from face object
                    face_embedding = face.embedding

                    if face_embedding is not None:
                        matched_student, similarity = self.find_matching_student(
                            face_embedding)

                        # Draw bounding box and name
                        bbox = face.bbox.astype(int)
                        if matched_student:
                            # Draw green box for matched faces
                            cv2.rectangle(
                                frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                            cv2.putText(frame, f"{matched_student} ({similarity:.2f})",
                                        (bbox[0], bbox[1] -
                                         10), cv2.FONT_HERSHEY_SIMPLEX,
                                        0.9, (0, 255, 0), 2)
                        else:
                            # Draw red box for unknown faces
                            cv2.rectangle(
                                frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
                            cv2.putText(frame, "Unknown", (bbox[0], bbox[1] - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

                # Calculate and display FPS
                fps_counter += 1
                if time.time() - prev_time > 1.0:
                    fps_display = fps_counter
                    fps_counter = 0
                    prev_time = time.time()

                cv2.putText(frame, f"FPS: {fps_display}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Write the frame if output path is provided
                if out is not None:
                    out.write(frame)

                # Display the frame if requested
                if display:
                    try:
                        cv2.imshow('Face Recognition', frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                    except cv2.error:
                        print(
                            "Warning: Display not available. Processing will continue without visual output.")
                        display = False

        finally:
            # Clean up
            cap.release()
            if out is not None:
                out.release()
            if display:
                cv2.destroyAllWindows()


# Example usage
if __name__ == "__main__":
    # Initialize the system
    face_system = FaceRecognitionSystem("dataset")

    # Process video without display
    face_system.process_video("input.mp4", "output.mp4", display=False)
