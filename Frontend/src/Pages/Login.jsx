import { useState } from "react";
import { useNavigate } from "react-router-dom";
import "./Login.css";
import login_back from "../assets/login-back.jpg";

function Login() {
  const [username, setUsername] = useState("");
  const [facultyId, setFacultyId] = useState("");
  const [password, setPassword] = useState("");
  const navigate = useNavigate();

  const handleLogin = () => {
    if (!facultyId) {
      alert("Faculty ID is required");
      return;
    }

    if (username === "admin" && password === "password") {
      navigate("/home");
    } else {
      alert("Invalid credentials");
    }
  };

  const handleSignUp = () => {
    navigate("/signup");
  };

  return (
    <div className="login-container" style={{ backgroundImage: `url(${login_back})` }}>
      <div className="login-card">
        <h1>Login</h1>
        <input 
          type="text" 
          placeholder="Faculty ID" 
          value={facultyId} 
          onChange={(e) => setFacultyId(e.target.value)} 
          required 
        />
        <input 
          type="text" 
          placeholder="Username" 
          value={username} 
          onChange={(e) => setUsername(e.target.value)} 
        />
        <input 
          type="password" 
          placeholder="Password" 
          value={password} 
          onChange={(e) => setPassword(e.target.value)} 
        />
        <button onClick={handleLogin}>Login</button>
        <p>
          Don't have an account?{" "}
          <span className="signup-link" onClick={handleSignUp}>Sign Up</span>
        </p>
      </div>
    </div>
  );
}

export default Login;
