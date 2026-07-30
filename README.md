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

Run the Flask app after a trained model exists in `models/`:

```bash
python backend/app.py
```

Open `http://127.0.0.1:5000`, upload a leaf image, and review the top predictions with treatment recommendations. Predictions below 60% confidence are shown as uncertain.

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
