# Labelbox to YOLO11

## Introduction

Downloads a dataset from Labelbox and formats it for YOLO11 object detection training.

## Installation

### Using Poetry (Recommended)

```bash
# Clone the repository
git clone git@github.com:stridera/labelbox_to_yolo.git

# Install dependencies (Poetry will create a virtual environment automatically)
poetry install
```

### Using pip

```bash
# Clone the repository
git clone git@github.com:stridera/labelbox_to_yolo.git

# Create a virtual environment (optional)
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Create a .env file with the following variables

```
API_KEY=<your labelbox api key>
PROJECT_ID=<your labelbox project id>
DATASET_ID=<your labelbox dataset id>
```

## Usage

### Download and create the dataset

```bash
# Using Poetry
poetry run python save_dataset.py

# Using pip
python ./save_dataset.py
```

### Train the model

#### Option 1: Transfer Learning (Recommended for quick results)
Uses pre-trained COCO weights as a starting point:

```bash
# Using Poetry
poetry run yolo train data=datasets/Robotron/dataset.yaml model=yolo11n.pt epochs=50 imgsz=640 batch=16

# Using pip
yolo train data=datasets/Robotron/dataset.yaml model=yolo11n.pt epochs=50 imgsz=640 batch=16
```

#### Option 2: Train from Scratch (Using only your data)
Trains a completely new model without any pre-trained weights:

```bash
# Using Poetry
poetry run yolo train data=datasets/Robotron/dataset.yaml model=yolo11n.yaml epochs=100 imgsz=640 batch=16 pretrained=False

# Using pip
yolo train data=datasets/Robotron/dataset.yaml model=yolo11n.yaml epochs=100 imgsz=640 batch=16 pretrained=False
```

**Model options:**
- **Transfer Learning (*.pt files):** `yolo11n.pt`, `yolo11s.pt`, `yolo11m.pt`, `yolo11l.pt`, `yolo11x.pt`
- **From Scratch (*.yaml files):** `yolo11n.yaml`, `yolo11s.yaml`, `yolo11m.yaml`, `yolo11l.yaml`, `yolo11x.yaml`

**When to use each approach:**
- **Transfer Learning:** Faster training (50 epochs), good results with limited data
- **From Scratch:** Pure custom model, no external data influence, requires more epochs (100+)

Note: Results are saved to `runs/detect/train`, `runs/detect/train2`, etc.

### Test the model

```bash
# Using Poetry
poetry run yolo predict model=runs/detect/train/weights/best.pt source=datasets/Robotron/images/ imgsz=640 conf=0.25

# Using pip
yolo predict model=runs/detect/train/weights/best.pt source=datasets/Robotron/images/ imgsz=640 conf=0.25
```

Note: Replace `train` with your experiment name (e.g., `train2`, `train3`)

### Validate the model

```bash
# Using Poetry
poetry run yolo val model=runs/detect/train/weights/best.pt data=datasets/Robotron/dataset.yaml

# Using pip
yolo val model=runs/detect/train/weights/best.pt data=datasets/Robotron/dataset.yaml
```

## Detailed Training & Usage Guide

For comprehensive training options, advanced configurations, monitoring, and Python integration examples, see [MODEL_USAGE.md](MODEL_USAGE.md).
