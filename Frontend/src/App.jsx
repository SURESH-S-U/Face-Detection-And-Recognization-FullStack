
import { BrowserRouter as Router, Routes, Route, Link } from "react-router-dom";
import Login from "./Pages/Login";
import Live_video from "./Pages/Live_video";
import Recorded_video from "./Pages/Recorded_video";
import Signup from "./Pages/Signup";
import Home from "./Pages/Home";
import "./App.css";

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Login />} />
        <Route path="/signup" element={<Signup/>} />
        <Route path="/home" element={<Home/>} />
        <Route path="/live" element={<Live_video />} />
        <Route path="/recorded" element={<Recorded_video />} />
      </Routes>
    </Router>
  );
}

export default App;