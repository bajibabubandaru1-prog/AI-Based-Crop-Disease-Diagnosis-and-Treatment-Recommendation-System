# AI-Based Crop Disease Diagnosis and Treatment Recommendation System

## Project Overview

This project aims to develop an AI-powered crop disease diagnosis and treatment recommendation system using Deep Learning and Computer Vision. The system identifies crop diseases from leaf images and recommends suitable organic treatments, chemical treatments, and preventive measures through a web application.

## Technologies

- Python
- TensorFlow / Keras
- OpenCV
- Flask
- MobileNetV2
- EfficientNetB0

## Dataset

- PlantVillage Dataset
- Local training path: `dataset/plantvillage/archive/PlantVillage`
- Expected folders: `train/` and `val/`, with one class folder per crop disease.
- The dataset is ignored by Git because image files are large.

## Machine Learning Pipeline

Install dependencies:

```bash
pip install -r requirements.txt
```

Train the first MobileNetV2 transfer learning model:

```bash
python training/train.py
```

The training script runs two phases:

- Frozen MobileNetV2 feature extraction.
- Fine-tuning with the last MobileNetV2 layers unfrozen at a lower learning rate.

Train EfficientNetB0 for comparison:

```bash
python training/train.py --model efficientnetb0
```

Resume EfficientNetB0 training from the latest completed epoch:

```bash
python training/train.py --model efficientnetb0 --resume
```

The resume checkpoint is saved after each epoch in `models/training_checkpoints/`.
If training stops in the middle of an epoch, rerunning with `--resume` continues from
the last fully completed epoch.
Progress details are written to
`models/training_checkpoints/efficientnetb0_resume_state.json`, and epoch metrics are
logged to `models/training_checkpoints/efficientnetb0_training_log.csv`.

Continue training from an already saved EfficientNetB0 model when no resume checkpoint
exists:

```bash
python training/train.py --model efficientnetb0 --continue-from-saved-model
```

If you know the saved model had already completed some epochs, pass that number:

```bash
python training/train.py --model efficientnetb0 --continue-from-saved-model --initial-epoch 5
```

Run more gently on a laptop while doing other work:

```bash
python training/train.py --model efficientnetb0 --resume --laptop-mode
```

Or control it manually:

```bash
python training/train.py --model efficientnetb0 --resume --batch-size 8 --cpu-threads 2
```

If pretrained EfficientNetB0 weights cannot be downloaded, train it from scratch:

```bash
python training/train.py --model efficientnetb0 --weights none --fine-tune-epochs 0
```

Evaluate the trained model:

```bash
python training/evaluate.py
```

Evaluate EfficientNetB0:

```bash
python training/evaluate.py --model efficientnetb0
```

Predict one leaf image:

```bash
python training/predict.py path/to/leaf_image.jpg
```

## Web Application

The Flask app provides a responsive upload interface for crop-leaf diagnosis. It shows
the most likely condition, top three matching classes, a confidence warning, and
organic, chemical, and preventive guidance.

It supports three image sources:

- Upload a JPG or PNG from the device.
- Paste a direct, public JPG or PNG image URL from the web.
- Use the mobile camera scanner to capture a leaf inside a scanning frame.

Camera capture requires HTTPS on most mobile browsers. Standard image upload works when
the phone opens the app through the same Wi-Fi or mobile hotspot network.

### Before running

At least one trained model must be present in `models/`:

- `crop_disease_mobilenetv2.keras` for MobileNetV2
- `crop_disease_efficientnetb0.keras` for EfficientNetB0

`models/class_names.txt` must also be present. The model files are intentionally
ignored by Git because they are large.

### Run locally

Use Python 3.12. Create a fresh virtual environment on every computer; do not copy a
`.venv` folder from another machine.

```powershell
git clone -b main https://github.com/bajibabubandaru1-prog/AI-Based-Crop-Disease-Diagnosis-and-Treatment-Recommendation-System.git
cd AI-Based-Crop-Disease-Diagnosis-and-Treatment-Recommendation-System
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Copy the trained models from the original computer into the cloned project's `models/`
folder. At minimum, copy `crop_disease_mobilenetv2.keras`. Copying
`crop_disease_efficientnetb0.keras` enables comparison, and `class_names.txt` must be
present with the models. Model files are intentionally excluded from Git because they
are large.

Check the installation before starting the app:

```powershell
python scripts\verify_setup.py
```

When the check reports only `OK`, start the app:

```powershell
python backend\app.py
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in a browser. Upload a clear,
well-lit JPG or PNG leaf image no larger than 10 MB. The first prediction can be
slower because TensorFlow loads the selected model into memory.

### App API

- `GET /api/health` reports class count and which trained models are available.
- `GET /api/classes` returns the supported PlantVillage classes.
- `POST /api/predict` accepts multipart form data with `image` and optional `model`
  (`mobilenetv2` or `efficientnetb0`). It returns the top three predictions and guidance.
- `POST /api/predict-url` accepts a public direct image URL in `image_url` and an
  optional model. The server downloads and validates the image before predicting.

Predictions below 60% confidence are treated as uncertain. They should be confirmed
with clearer imagery or an agricultural professional, especially before any treatment.

## Team Members

- Baji Babu Bandaru
- Naga Niranjan Reddy Mandha
- Pranathi Nooka
- Tasleem Fathima

## Repository Structure

```
backend/
dataset/
docs/
frontend/
knowledge_base/
models/
notebooks/
tests/
training/
```

## Status

 Project  completed.
