const form = document.querySelector("#prediction-form");
const resultPanel = document.querySelector("#result");
const preview = document.querySelector("#preview");
const statusText = document.querySelector("#status");
const predictionText = document.querySelector("#prediction");
const confidenceText = document.querySelector("#confidence");
const topPredictions = document.querySelector("#top-predictions");
const organicText = document.querySelector("#organic");
const chemicalText = document.querySelector("#chemical");
const preventionText = document.querySelector("#prevention");

function formatClassName(className) {
  return className.replaceAll("___", " - ").replaceAll("_", " ");
}

function formatPercent(value) {
  return `${(value * 100).toFixed(2)}%`;
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const button = form.querySelector("button");
  const formData = new FormData(form);

  button.disabled = true;
  button.textContent = "Predicting...";

  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      body: formData,
    });
    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.error || "Prediction failed.");
    }

    const top = data.top_prediction;
    const recommendation = data.recommendation;

    resultPanel.classList.remove("hidden");
    preview.src = data.image_url;
    predictionText.textContent = recommendation.display_name || formatClassName(top.class_name);
    confidenceText.textContent = `Confidence: ${formatPercent(top.confidence)}`;

    statusText.className = data.is_confident ? "status good" : "status warning";
    statusText.textContent = data.is_confident
      ? "Prediction confidence is acceptable."
      : `Prediction is uncertain below ${formatPercent(data.confidence_threshold)}. Try a clearer close-up leaf image.`;

    topPredictions.innerHTML = data.top_predictions
      .map((item) => `<div>${formatClassName(item.class_name)} - ${formatPercent(item.confidence)}</div>`)
      .join("");

    organicText.textContent = recommendation.organic_treatment;
    chemicalText.textContent = recommendation.chemical_treatment;
    preventionText.textContent = recommendation.prevention;
  } catch (error) {
    resultPanel.classList.remove("hidden");
    preview.removeAttribute("src");
    statusText.className = "status warning";
    statusText.textContent = error.message;
    predictionText.textContent = "";
    confidenceText.textContent = "";
    topPredictions.innerHTML = "";
    organicText.textContent = "";
    chemicalText.textContent = "";
    preventionText.textContent = "";
  } finally {
    button.disabled = false;
    button.textContent = "Predict Disease";
  }
});
