import { useState, useEffect } from "react";
import "./Live_video.css";

function Live_video() {
  const [isCameraOn, setIsCameraOn] = useState(false);
  const [rollNumbers, setRollNumbers] = useState([
    { roll_number: "21CSE101", timestamp: "2025-02-12 10:15:30" },
    { roll_number: "21CSE102", timestamp: "2025-02-12 10:16:05" },
    { roll_number: "21CSE103", timestamp: "2025-02-12 10:17:20" }
  ]);
  const [videoSrc, setVideoSrc] = useState(null);

  useEffect(() => {
    if (isCameraOn) {
      setVideoSrc("http://127.0.0.1:8000/video_feed"); // Backend API URL for streaming
    } else {
      setVideoSrc(null);
    }
  }, [isCameraOn]);

  return (
    <div className="video-container">

      {/* Left Side - Video Section */}
      <div className="left-side">
        <h1>Live Detection</h1>
        <div className="video-slot">
          {isCameraOn ? <img src={videoSrc} alt="Live Video Feed" /> : "Camera Off"}
        </div>

        <div className="button-group">
          <button onClick={() => setIsCameraOn(true)}>Turn On Camera</button>
          <button onClick={() => setIsCameraOn(false)}>Stop Finding</button>
        </div>
      </div>

      {/* Right Side - Detected Faces Table */}
        <div className="info-slot">
          <h2>Detected Faces</h2>
          <table>
            <thead>
              <tr>
                <th>Roll Number</th>
                <th>Timestamp</th>
              </tr>
            </thead>
            <tbody>
              {rollNumbers.map((face, index) => (
                <tr key={index}>
                  <td>{face.roll_number}</td>
                  <td>{face.timestamp}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
    </div>
  );
}

export default Live_video;
