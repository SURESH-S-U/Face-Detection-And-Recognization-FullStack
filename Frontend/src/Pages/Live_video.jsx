import { useState, useEffect } from "react";
import axios from "axios";
import "./Live_video.css";

function Live_video() {
  const [isCameraOn, setIsCameraOn] = useState(false);
  const [detectedFaces, setDetectedFaces] = useState([]);
  const [videoSrc, setVideoSrc] = useState(null);
  const backendUrl = "http://localhost:8000";

  useEffect(() => {
    if (isCameraOn) {
      setVideoSrc(`${backendUrl}/video_feed`);
    } else {
      setVideoSrc(null);
    }
  }, [isCameraOn]);

  useEffect(() => {
    if (isCameraOn) {
      const fetchFaces = () => {
        axios.get(`${backendUrl}/detected_faces`)
          .then(response => {
            console.log("API Response:", response.data); // Debugging API response
            if (response.data && response.data.faces) {
              setDetectedFaces(response.data.faces);
              console.log("Updated State (Detected Faces):", response.data.faces);
            } else {
              console.warn("Unexpected API response format:", response.data);
            }
          })
          .catch(err => console.error("Error fetching faces:", err));
      };

      const interval = setInterval(fetchFaces, 3000);
      return () => clearInterval(interval);
    }
  }, [isCameraOn]);

  return (
    <div className="video-container">
      <div className="left-side">
        <h1>Live Detection</h1>
        <div className="video-slot">
          {isCameraOn && videoSrc ? (
            <img
              src={videoSrc}
              alt="Live Video Feed"
              style={{ width: "100%", height: "100%", objectFit: "cover" }}
              onError={(e) => {
                console.error("Error loading video feed", e);
                alert("Error loading video feed. Check backend service.");
              }}
            />
          ) : (
            <p>Camera Off</p>
          )}
        </div>
        <div className="button-group">
          <button onClick={() => setIsCameraOn(true)}>Turn On Camera</button>
          <button onClick={() => setIsCameraOn(false)}>Stop Finding</button>
        </div>
      </div>
      <div className="info-slot">
        <h2>Detected Faces</h2>
        <table>
          <thead>
            <tr>
              <th>Name</th>
              <th>Timestamp</th>
            </tr>
          </thead>
          <tbody>
            {detectedFaces.length > 0 ? (
              detectedFaces.map((face, index) => (
                <tr key={index}>
                  <td>{face.name || "Unknown"}</td>
                  <td>{face.timestamp || "No Timestamp"}</td>
                </tr>
              ))
            ) : (
              <tr>
                <td colSpan="2">No faces detected</td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export default Live_video; 
