const emailInput = document.getElementById("emailInput");
const checkButton = document.getElementById("checkButton");
const prediction = document.getElementById("prediction");
const confidence = document.getElementById("confidence");
const resultText = document.querySelector(".result-text");

// Backend API
const API_URL = "http://127.0.0.1:5000/predict";

checkButton.addEventListener("click", async () => {

    const email = emailInput.value.trim();

    if (email === "") {
        alert("Please enter an email.");
        return;
    }

    prediction.textContent = "Checking...";
    confidence.textContent = "--";
    resultText.textContent = "Analyzing your email...";

    checkButton.disabled = true;
    checkButton.textContent = "Checking...";

    try {

        const response = await fetch(API_URL, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                email: email
            })
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || "Something went wrong");
        }

        prediction.textContent = data.prediction;
        confidence.textContent = data.confidence + "%";

        if (data.prediction.toLowerCase() === "spam") {
            resultText.textContent = "⚠️ This email appears to be Spam.";
        } else {
            resultText.textContent = "✅ This email appears to be Not Spam.";
        }

    } catch (error) {

        console.error("Error:", error);

        prediction.textContent = "Error";
        confidence.textContent = "--";
        resultText.textContent = "Unable to connect to the server.";

        alert("Could not connect to the Flask backend.");

    } finally {

        checkButton.disabled = false;
        checkButton.textContent = "Check Email";
    }
});
