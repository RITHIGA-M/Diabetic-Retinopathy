// Import Firebase SDK (Modular Approach)
import { initializeApp } from "https://www.gstatic.com/firebasejs/9.6.1/firebase-app.js";
import { getAuth, createUserWithEmailAndPassword, signInWithEmailAndPassword, signOut, updateProfile, onAuthStateChanged } from "https://www.gstatic.com/firebasejs/9.6.1/firebase-auth.js";

// Firebase Configuration
const firebaseConfig = {
    apiKey: "AIzaSyBUc09vVr13lufna3PcRKIvkScJMmL84F8",
    authDomain: "diabetic-retinopathy-d9ad0.firebaseapp.com",
    databaseURL: "https://diabetic-retinopathy-d9ad0-default-rtdb.firebaseio.com",
    projectId: "diabetic-retinopathy-d9ad0",
    storageBucket: "diabetic-retinopathy-d9ad0.appspot.com",
    messagingSenderId: "784479554899",
    appId: "1:784479554899:web:741bb70f664687c789e5a6",
    measurementId: "G-T9E5GHPJ65"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);
const auth = getAuth(app);

// User Signup with Name
async function signup() {
    let name = document.getElementById("signup-name").value;
    let email = document.getElementById("signup-email").value;
    let password = document.getElementById("signup-password").value;

    try {
        const userCredential = await createUserWithEmailAndPassword(auth, email, password);
        const user = userCredential.user;

        // Update profile with name
        await updateProfile(user, { displayName: name });

        alert("User Created Successfully with Name!");
        window.location.href = "dashboard.html"; // Redirect after signup
    } catch (error) {
        document.getElementById("signup-error").innerText = error.message;
    }
}

// User Login
async function login() {
    let email = document.getElementById("login-email").value;
    let password = document.getElementById("login-password").value;

    try {
        const userCredential = await signInWithEmailAndPassword(auth, email, password);
        const user = userCredential.user;

        alert("Login Successful! Welcome, " + (user.displayName || "User"));
        window.location.href = "dashboard.html"; // Redirect after login
    } catch (error) {
        document.getElementById("login-error").innerText = error.message;
    }
}

// Logout
async function logout() {
    try {
        await signOut(auth);
        alert("Logged Out Successfully!");
        window.location.href = "index.html"; // Redirect to login page
    } catch (error) {
        alert(error.message);
    }
}

// Check Auth State (Keep Users Logged In)
onAuthStateChanged(auth, (user) => {
    if (user) {
        console.log("User logged in:", user.displayName || user.email);
        document.getElementById("user-info").innerText = "Welcome, " + (user.displayName || "User");
    } else {
        console.log("User is logged out");
    }
});

// Expose functions globally (needed for inline onclick handlers)
window.signup = signup;
window.login = login;
window.logout = logout;
