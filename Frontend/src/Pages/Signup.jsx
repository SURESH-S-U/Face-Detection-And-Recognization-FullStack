import { useState } from "react";
import { useNavigate } from "react-router-dom";
import "./SignUp.css";
import signup_back from "../assets/login-back.jpg"; 

function SignUp() {
  const [facultyId, setFacultyId] = useState("");
  const [username, setUsername] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const navigate = useNavigate();

  const handleSignUp = () => {
    if (!facultyId) {
      alert("Faculty ID is required");
      return;
    }
    if (!username || !email || !password) {
      alert("All fields are required");
      return;
    }

    // Simulate a sign-up process (you can replace this with API integration)
    alert("Sign-up successful! Redirecting to login...");
    navigate("/");
  };

  return (
    <div className="signup-container" style={{ backgroundImage: `url(${signup_back})` }}>
      <div className="signup-card">
        <h1>Sign Up</h1>
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
          type="email" 
          placeholder="Email" 
          value={email} 
          onChange={(e) => setEmail(e.target.value)} 
        />
        <input 
          type="password" 
          placeholder="Password" 
          value={password} 
          onChange={(e) => setPassword(e.target.value)} 
        />
        <button onClick={handleSignUp}>Sign Up</button>
        <p>
          Already have an account?{" "}
          <span className="signin-link" onClick={() => navigate("/")}>Loging In</span>
        </p>
      </div>
    </div>
  );
}

export default SignUp;
