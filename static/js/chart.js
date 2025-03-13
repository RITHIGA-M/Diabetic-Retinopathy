

firebaseConfig = {
    "apiKey": "AIzaSyBUc09vVr13lufna3PcRKIvkScJMmL84F8",
    "authDomain": "diabetic-retinopathy-d9ad0.firebaseapp.com",
    "databaseURL": "https://diabetic-retinopathy-d9ad0-default-rtdb.firebaseio.com",
    "projectId": "diabetic-retinopathy-d9ad0",
    "storageBucket": "diabetic-retinopathy-d9ad0.appspot.com",
    "messagingSenderId": "784479554899",
    "appId": "1:784479554899:web:741bb70f664687c789e5a6",
    "measurementId": "G-T9E5GHPJ65"
}

firebase.initializeApp(firebaseConfig);
const db = firebase.firestore();

// Function to Fetch Patient Risk Scores
function fetchRiskScores() {
    db.collection("patients").orderBy("lastUpdated", "desc").onSnapshot((querySnapshot) => {
        let tableBody = document.querySelector("#riskTable tbody");
        tableBody.innerHTML = "";

        let riskData = [];

        querySnapshot.forEach((doc) => {
            let data = doc.data();
            let alertMsg = data.riskScore > 70 ? "<span class='alert'>High Risk!</span>" : "Normal";

            let row = `
                <tr>
                    <td>${data.name}</td>
                    <td>${data.riskScore}</td>
                    <td>${new Date(data.lastUpdated.seconds * 1000).toLocaleDateString()}</td>
                    <td>${alertMsg}</td>
                </tr>
            `;

            tableBody.innerHTML += row;

            // Store data for chart
            riskData.push({
                name: data.name,
                riskScore: data.riskScore,
                date: new Date(data.lastUpdated.seconds * 1000).toLocaleDateString()
            });
        });

        // Update chart
        updateRiskChart(riskData);
    });
}

// Function to Update Risk Progression Chart
function updateRiskChart(riskData) {
    let labels = riskData.map(d => d.date);
    let scores = riskData.map(d => d.riskScore);

    let ctx = document.getElementById('riskChart').getContext('2d');

    if (window.riskChartInstance) {
        window.riskChartInstance.destroy();
    }

    window.riskChartInstance = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: 'Risk Score Progression',
                data: scores,
                borderColor: 'red',
                fill: false
            }]
        },
        options: {
            responsive: true,
            scales: {
                y: { beginAtZero: true }
            }
        }
    });
}

// Call Function on Page Load
fetchRiskScores();
