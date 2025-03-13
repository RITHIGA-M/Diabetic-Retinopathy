function uploadAndPredict() {
    const name = document.getElementById("name").value;
    const fileInput = document.getElementById("file");

    if (!name || !fileInput.files.length) {
        alert("Please enter a name and select an image.");
        return;
    }

    const file = fileInput.files[0];
    const formData = new FormData();
    formData.append("name", name);
    formData.append("file", file);

    fetch("/predict", {
        method: "POST",
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            alert(data.error);
            return;
        }

        document.getElementById("result").innerText = `Prediction: ${data.prediction} (Confidence: ${data.confidence.toFixed(2)}%)`;
        document.getElementById("uploadedImage").src = data.image_url;
        document.getElementById("result-container").style.display = "block";
        document.getElementById("downloadReport").style.display = "block";
    })
    .catch(error => console.error("Error:", error));
}

function downloadReport() {
    const name = document.getElementById("name").value;
    const resultText = document.getElementById("result").innerText;
    const match = resultText.match(/Prediction: (.*?) \(Confidence: (.*?)%\)/);

    if (!match) {
        alert("Invalid result format.");
        return;
    }

    const [_, diagnosis, confidence] = match;
    window.location.href = `/download_report?name=${name}&diagnosis=${diagnosis}&confidence=${confidence}`;
}
