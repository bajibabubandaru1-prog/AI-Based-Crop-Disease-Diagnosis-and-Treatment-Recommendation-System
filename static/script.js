const form = document.querySelector("#prediction-form");
const imageInput = document.querySelector("#image");
const imageUrlInput = document.querySelector("#image-url");
const dropzone = document.querySelector("#dropzone");
const sourceTabs = document.querySelectorAll(".source-tab");
const sourcePanels = document.querySelectorAll(".source-panel");
const selectedFile = document.querySelector("#selected-file");
const selectedFileName = document.querySelector("#selected-file-name");
const clearImage = document.querySelector("#clear-image");
const button = document.querySelector("#predict-button");
const formMessage = document.querySelector("#form-message");
const resultEmpty = document.querySelector("#result-empty");
const resultContent = document.querySelector("#result-content");
const preview = document.querySelector("#preview");
const predictionText = document.querySelector("#prediction");
const modelUsed = document.querySelector("#model-used");
const confidenceBadge = document.querySelector("#confidence-badge");
const statusText = document.querySelector("#status");
const topPredictions = document.querySelector("#top-predictions");
const topPredictionsHeading = document.querySelector("#top-predictions-heading");
const organicText = document.querySelector("#organic");
const chemicalText = document.querySelector("#chemical");
const preventionText = document.querySelector("#prevention");
const cameraPreview = document.querySelector("#camera-preview");
const cameraCanvas = document.querySelector("#camera-canvas");
const cameraPlaceholder = document.querySelector("#camera-placeholder");
const openCameraButton = document.querySelector("#open-camera");
const captureButton = document.querySelector("#capture-image");

let activeMode = "upload";
let cameraStream = null;

function formatPercent(value) { return `${(value * 100).toFixed(1)}%`; }
function setMessage(message = "") { formMessage.textContent = message; }

function stopCamera() {
  if (cameraStream) cameraStream.getTracks().forEach((track) => track.stop());
  cameraStream = null;
  cameraPreview.srcObject = null;
  captureButton.disabled = true;
  cameraPlaceholder.hidden = false;
}

function setSelectedFile(file) {
  if (!file) {
    selectedFile.hidden = true;
    selectedFileName.textContent = "";
    dropzone.classList.remove("has-file");
    return;
  }
  selectedFile.hidden = false;
  selectedFileName.textContent = file.name;
  dropzone.classList.add("has-file");
  setMessage("");
}

function resetSelectedFile() { imageInput.value = ""; setSelectedFile(null); }

function setMode(mode) {
  activeMode = mode;
  sourceTabs.forEach((tab) => {
    const selected = tab.dataset.mode === mode;
    tab.classList.toggle("active", selected);
    tab.setAttribute("aria-selected", String(selected));
  });
  sourcePanels.forEach((panel) => { panel.hidden = panel.dataset.panel !== mode; });
  if (mode !== "scan") stopCamera();
  setMessage("");
}

async function openCamera() {
  if (!navigator.mediaDevices?.getUserMedia) {
    setMessage("Camera scanning needs HTTPS on mobile. Use upload for this local test, then use an HTTPS deployment for scanning.");
    return;
  }
  try {
    cameraStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: { ideal: "environment" } }, audio: false });
    cameraPreview.srcObject = cameraStream;
    cameraPlaceholder.hidden = true;
    captureButton.disabled = false;
    setMessage("");
  } catch (error) {
    setMessage("Camera access was not available. Allow camera permission, or use image upload instead.");
  }
}

function captureCameraImage() {
  if (!cameraStream || !cameraPreview.videoWidth) return;
  cameraCanvas.width = cameraPreview.videoWidth;
  cameraCanvas.height = cameraPreview.videoHeight;
  cameraCanvas.getContext("2d").drawImage(cameraPreview, 0, 0, cameraCanvas.width, cameraCanvas.height);
  cameraCanvas.toBlob((blob) => {
    if (!blob) return;
    const file = new File([blob], `leaf-scan-${Date.now()}.jpg`, { type: "image/jpeg" });
    const transfer = new DataTransfer();
    transfer.items.add(file);
    imageInput.files = transfer.files;
    setSelectedFile(file);
    stopCamera();
    setMode("upload");
    setMessage("Scan captured. Select Analyze leaf to get the diagnosis.");
  }, "image/jpeg", 0.92);
}

imageInput.addEventListener("change", () => setSelectedFile(imageInput.files[0]));
clearImage.addEventListener("click", resetSelectedFile);
sourceTabs.forEach((tab) => tab.addEventListener("click", () => setMode(tab.dataset.mode)));
openCameraButton.addEventListener("click", openCamera);
captureButton.addEventListener("click", captureCameraImage);

["dragenter", "dragover"].forEach((eventName) => dropzone.addEventListener(eventName, (event) => { event.preventDefault(); dropzone.classList.add("dragging"); }));
["dragleave", "drop"].forEach((eventName) => dropzone.addEventListener(eventName, (event) => { event.preventDefault(); dropzone.classList.remove("dragging"); }));
dropzone.addEventListener("drop", (event) => {
  const [file] = event.dataTransfer.files;
  if (!file) return;
  const transfer = new DataTransfer();
  transfer.items.add(file);
  imageInput.files = transfer.files;
  setSelectedFile(file);
});

function showResult(data) {
  const top = data.top_prediction;
  const recommendation = data.recommendation;
  const accepted = data.diagnosis_accepted;
  resultEmpty.hidden = true;
  resultContent.hidden = false;
  preview.src = data.image_url;
  predictionText.textContent = recommendation.display_name || top.display_name;
  modelUsed.textContent = `Analyzed with ${data.model_label}`;
  confidenceBadge.className = `confidence-badge ${accepted ? "high" : "low"}`;
  confidenceBadge.textContent = accepted ? `${formatPercent(top.confidence)} confidence` : "Not accepted";
  statusText.className = `status ${accepted ? "good" : "warning"}`;
  statusText.textContent = accepted
    ? "This result is above the confidence threshold. Inspect the leaf and follow the guidance below."
    : `No disease diagnosis was accepted because the closest model match is below the ${formatPercent(data.confidence_threshold)} confidence threshold. This can happen with random, non-leaf, or unclear images.`;
  topPredictionsHeading.textContent = accepted ? "Alternative matches" : "Closest model matches (not a diagnosis)";
  topPredictions.replaceChildren(...data.top_predictions.map((item) => {
    const row = document.createElement("div"); row.className = "prediction-row";
    const label = document.createElement("span"); label.textContent = item.display_name;
    const value = document.createElement("strong"); value.textContent = formatPercent(item.confidence);
    const meter = document.createElement("span"); meter.className = "prediction-meter";
    const fill = document.createElement("span"); fill.style.width = `${Math.max(item.confidence * 100, 2)}%`;
    meter.append(fill); row.append(label, value, meter); return row;
  }));
  organicText.textContent = recommendation.organic_treatment;
  chemicalText.textContent = recommendation.chemical_treatment;
  preventionText.textContent = recommendation.prevention;
  resultContent.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const useUrl = activeMode === "link";
  if (useUrl && !imageUrlInput.value.trim()) { setMessage("Paste a direct public image link before analyzing."); imageUrlInput.focus(); return; }
  if (!useUrl && !imageInput.files[0]) { setMessage("Choose a JPG or PNG leaf image before analyzing."); imageInput.focus(); return; }
  button.disabled = true;
  button.querySelector("span").textContent = "Analyzing...";
  setMessage("");
  try {
    const endpoint = useUrl ? "/api/predict-url" : "/api/predict";
    const response = await fetch(endpoint, { method: "POST", body: new FormData(form) });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Prediction failed. Please try again.");
    showResult(data);
  } catch (error) { setMessage(error.message); }
  finally { button.disabled = false; button.querySelector("span").textContent = "Analyze leaf"; }
});
