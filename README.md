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

Evaluate the trained model:

```bash
python training/evaluate.py
```

Predict one leaf image:

```bash
python training/predict.py path/to/leaf_image.jpg
```

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

🚧 Project initialization completed. Development is in progress.
