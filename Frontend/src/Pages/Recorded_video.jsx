import { useState } from "react";
import "./Recorded_video.css";

function Recorded_video() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [processedVideo, setProcessedVideo] = useState(null);
  const [rollNumbers, setRollNumbers] = useState([
    { roll_number: "21BCS001", timestamp: "00:00:10" },
    { roll_number: "21BCS002", timestamp: "00:01:20" },
    { roll_number: "21BCS003", timestamp: "00:02:45" },
  ]);

  const handleFileChange = (event) => {
    setSelectedFile(event.target.files[0]);
  };

  const handleUpload = async () => {
    if (!selectedFile) {
      alert("Please select a video file first!");
      return;
    }

    setIsProcessing(true);

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const response = await fetch("http://127.0.0.1:8000/upload_video/", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();
      if (data.video_url) {
        setProcessedVideo("http://127.0.0.1:8000" + data.video_url);
      }
    } catch (error) {
      console.error("Error uploading video:", error);
    }
  };

  const handleStop = () => {
    setIsProcessing(false);
    setProcessedVideo(null);
    setRollNumbers([]);
  };

  return (
    <div className="Rvideo-container">
      <div className="Rleft-side">
        <h1>Recorded Video Analysis</h1>
        <div className="Rvideo-slot">
          {isProcessing ? (
            processedVideo ? (
              <video controls autoPlay>
                <source src={processedVideo} type="video/mp4" />
                Your browser does not support the video tag.
              </video>
            ) : (
              "Processing Video..."
            )
          ) : (
            "Upload a Video"
          )}
        </div>
        <input className="choose-file" type="file" accept="video/*" onChange={handleFileChange} />
        <div className="Rbutton-group">
          <button onClick={handleUpload} disabled={isProcessing}>
            Upload Video
          </button>
          <button onClick={handleStop}>Stop Finding</button>
        </div>
      </div>
      <div className="Rinfo-slot">
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

export default Recorded_video;