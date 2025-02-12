import React from "react";
import { useNavigate } from "react-router-dom";
import "./Home.css";
import logo from "../assets/logo.png";
import { FaUser } from "react-icons/fa";
import faceImage from "../assets/face-image.jpg";

const Home = () => {
  const navigate = useNavigate();

  return (
    <div className="home-container">
      {/* Left Section */}
      <div className="left-section">
        <img src={logo} alt="Logo" className="logo" />
        <div className="user-icon"><FaUser size={30} /></div>
        <h1>Welcome to Face Identification System</h1>
        <p>
          Our system provides advanced face detection and recognition for various applications.
        </p>
        <div className="buttons">
          <button className="btn live-detection" onClick={() => navigate("/live")}>
            Live Detection
          </button>
          <button className="btn recorded-video" onClick={() => navigate("/recorded")   }>
            Recorded Video
          </button>
        </div>
      </div>

      {/* Right Section */}
      <div className="right-section">
        <img src={faceImage} alt="Face Identification" className="main-image" />
      </div>
    </div>
  );
};

export default Home;
