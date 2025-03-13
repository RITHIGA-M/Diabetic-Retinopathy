import { initializeApp } from "https://www.gstatic.com/firebasejs/9.6.10/firebase-app.js";
import { getDatabase, ref, set, get } from "https://www.gstatic.com/firebasejs/9.6.10/firebase-database.js";

// ✅ Firebase Configuration
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

// ✅ Initialize Firebase
const app = initializeApp(firebaseConfig);
const db = getDatabase(app);

// ✅ Save Patient Data
function savePatientData(name, email, condition) {
    set(ref(db, "Patients/" + name), {
        email: email,
        condition: condition
    }).then(() => {
        console.log("Data saved successfully.");
    }).catch((error) => {
        console.error("Error saving data:", error);
    });
}

// ✅ Get Patient Data
function getPatientData(name) {
    get(ref(db, "Patients/" + name)).then((snapshot) => {
        if (snapshot.exists()) {
            console.log(snapshot.val());
        } else {
            console.log("No patient data found!");
        }
    }).catch((error) => {
        console.error("Error fetching data:", error);
    });
}

// ✅ Example Usage
savePatientData("Rithiga", "rithiga43@gmail.com", "Diabetic Retinopathy");
getPatientData("Rithiga");
