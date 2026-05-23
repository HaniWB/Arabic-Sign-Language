# Arabic Sign Language - Run Guide

This repository contains two main experiment tracks:

1. YOLOv5 object detection (class + bounding box)
2. ANN classification baseline (class only)

It also includes a Flask app for live webcam inference.

## 1) Prerequisites

- Windows + PowerShell
- Python 3.10+
- NVIDIA GPU (optional, but recommended)

## 2) Project Layout

- `arabic_sign_language/deep learning/`
  - `app.py` (Flask app)
  - `yolov5/` (original YOLO code + runs)
- `arabic_sign_language/02_deep_learning/yolov5/`
  - modified YOLO copy
  - `train_ann_classifier.py` (ANN training)
- `arabic_sign_language/Arabic sign language translator.v3i.yolov5pytorch/`
  - dataset (`train/`, `valid/`, `test/`, `data.yaml`)

## 3) Install Dependencies

From each YOLO folder where you will run code:

```powershell
cd "C:\Users\hanib\Downloads\Arabic-Sign-Language\arabic_sign_language\deep learning\yolov5"
python -m pip install -r requirements.txt
python -m pip install flask flask-cors
```

For ANN scripts (same environment):

```powershell
python -m pip install torch torchvision pillow pyyaml pandas matplotlib
```

## 4) Fix Dataset Path in data.yaml

Edit:

`C:\Users\hanib\Downloads\Arabic-Sign-Language\arabic_sign_language\Arabic sign language translator.v3i.yolov5pytorch\data.yaml`

Set:

```yaml
path: C:/Users/hanib/Downloads/Arabic-Sign-Language/arabic_sign_language/Arabic sign language translator.v3i.yolov5pytorch
train: train/images
val: valid/images
test: test/images
nc: 13
names: ['Dog', 'Hello', 'No', 'Thanks', 'fine', 'love', 'me', 'mother', 'smile', 'sorry', 'sunday', 'yes', 'you']
```

## 5) Run Flask App (Auto Webcam)

```powershell
cd "C:\Users\hanib\Downloads\Arabic-Sign-Language\arabic_sign_language\deep learning"
python .\app.py
```

Notes:

- Webcam detection auto-starts at app launch.
- App runs at `http://127.0.0.1:8080`.

## 6) Train/Validate Original YOLO (Detection)

```powershell
cd "C:\Users\hanib\Downloads\Arabic-Sign-Language\arabic_sign_language\deep learning\yolov5"
python train.py --weights yolov5s.pt --data "C:\Users\hanib\Downloads\Arabic-Sign-Language\arabic_sign_language\Arabic sign language translator.v3i.yolov5pytorch\data.yaml" --img 640 --batch 8 --epochs 500 --device 0
```

Evaluate on test split:

```powershell
python val.py --weights runs/train/exp14/weights/best.pt --data "C:\Users\hanib\Downloads\Arabic-Sign-Language\arabic_sign_language\Arabic sign language translator.v3i.yolov5pytorch\data.yaml" --img 640 --batch 8 --task test
```

## 7) Train ANN Baseline (Classification Only)

```powershell
cd "C:\Users\hanib\Downloads\Arabic-Sign-Language\arabic_sign_language\02_deep_learning\yolov5"
python .\train_ann_classifier.py --data-dir "C:\Users\hanib\Downloads\Arabic-Sign-Language\arabic_sign_language\Arabic sign language translator.v3i.yolov5pytorch" --epochs 50 --batch-size 32 --img-size 128 --patience 8 --out sign_ann_classifier_e50.pt
```

This script auto-converts YOLO dataset format into classification folders under:

- `_ann_cls_dataset/train/...`
- `_ann_cls_dataset/val/...`

## 8) Evaluate ANN

```powershell
python "C:\Users\hanib\Downloads\Arabic-Sign-Language\arabic_sign_language\deep learning\yolov5\eval_ann_classifier.py" --model "C:\Users\hanib\Downloads\Arabic-Sign-Language\arabic_sign_language\02_deep_learning\yolov5\sign_ann_classifier_e50.pt" --data-dir "C:\Users\hanib\Downloads\Arabic-Sign-Language\arabic_sign_language\Arabic sign language translator.v3i.yolov5pytorch"
```

## 9) ANN Prediction (Image/Webcam)

Image:

```powershell
python "C:\Users\hanib\Downloads\Arabic-Sign-Language\arabic_sign_language\deep learning\yolov5\predict_ann.py" --model "C:\Users\hanib\Downloads\Arabic-Sign-Language\arabic_sign_language\02_deep_learning\yolov5\sign_ann_classifier_e50.pt" --image "C:\FULL\PATH\to\image.jpg"
```

Webcam:

```powershell
python "C:\Users\hanib\Downloads\Arabic-Sign-Language\arabic_sign_language\deep learning\yolov5\predict_ann.py" --model "C:\Users\hanib\Downloads\Arabic-Sign-Language\arabic_sign_language\02_deep_learning\yolov5\sign_ann_classifier_e50.pt" --webcam
```

## 10) Comparison Notebook

Open:

`C:\Users\hanib\Downloads\Arabic-Sign-Language\arabic_sign_language\deep learning\yolov5\copmaresion.ipynb`

Run all cells to compare YOLO vs ANN automatically.

## 11) Common Issues

- `No module named flask`:
  - `python -m pip install flask flask-cors`
- Path errors with folders containing spaces:
  - Always wrap paths in quotes.
- Dataset not found:
  - Re-check `data.yaml` `path:` value.

