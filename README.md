# Face-Detection-FullStack

# Face Detection and Student Roll Number Recognition using YOLOv8

This project is a full stack application that uses a YOLOv8 object detection model for face detection via a surveillance camera and fetches students' roll numbers. The frontend is built with React, and the backend uses FastAPI with MongoDB as the database.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [Frontend Features](#frontend-features)
- [Backend Features](#backend-features)
- [Database Schema](#database-schema)
- [Model Training](#model-training)
- [Future Enhancements](#future-enhancements)
- [Contributors](#contributors)

---

## Project Overview
This application detects student faces using a YOLOv8 model from a live surveillance camera feed. Once a face is detected, the system matches it to a student roll number stored in the database. The FastAPI backend handles the API requests and database operations, while the React frontend provides an intuitive UI.

---

## Technologies Used
- **Frontend:** React, Tailwind CSS
- **Backend:** FastAPI, Python
- **Database:** MongoDB
- **Object Detection Model:** YOLOv8 (Ultralytics)
- **Tools:** VS Code, Docker (optional), Node.js, Python

---

## Project Structure
```
face-detection-yolo/
│
├── backend/
│   ├── main.py
│   ├── models/
│   ├── routes/
│   ├── services/
│   ├── database/
│   └── requirements.txt
│
├── frontend/
│   ├── public/
│   ├── src/
│   ├── components/
│   ├── pages/
│   ├── App.js
│   └── package.json
│
└── README.md
```

---

## Installation
### Backend Setup
1. Clone the repository.
2. Navigate to the `backend` directory.
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the FastAPI server:
   ```bash
   uvicorn main:app --reload
   ```

### Frontend Setup
1. Navigate to the `frontend` directory.
2. Install dependencies:
   ```bash
   npm install
   ```
3. Run the React development server:
   ```bash
   npm start
   ```

---

## Usage
- Access the frontend at `http://localhost:3000`.
- The camera feed will automatically start detecting faces.
- Detected faces will be sent to the backend for identification.
- Matched roll numbers will be displayed in the frontend UI.

---

## API Endpoints
- `POST /api/detect` - Receives images from the frontend and returns detected roll numbers.
- `GET /api/students` - Fetches student data from the database.
- `POST /api/students` - Adds new student data.

---

## Frontend Features
- Live camera feed display.
- Display of detected faces and corresponding roll numbers.
- User-friendly interface with Tailwind CSS.

---

## Backend Features
- YOLOv8 model integration for face detection.
- FastAPI endpoints for real-time communication.
- MongoDB integration for storing student data.

---

## Database Schema
- **students**
  - `_id`: ObjectId
  - `name`: String
  - `roll_number`: String
  - `face_embedding`: Array

---

## Model Training
- YOLOv8 model trained on a dataset of student faces.
- Augmented images for better accuracy.
- Exported model stored in `backend/models/`.

---

## Future Enhancements
- Implement attendance tracking.
- Add user authentication.
- Optimize face detection accuracy.

---

## Contributors
- **Suresh S U** – Full Stack Developer

---

> *This project is part of my AI and Data Science coursework at Bannari Amman Institute of Technology.*

