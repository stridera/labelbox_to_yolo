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

```bash
# Using Poetry
poetry run yolo train data=datasets/Robotron/dataset.yaml model=yolo11n.pt epochs=3 imgsz=640 batch=16

# Using pip
yolo train data=datasets/Robotron/dataset.yaml model=yolo11n.pt epochs=3 imgsz=640 batch=16
```

**Model options:**
- `yolo11n.pt` - Nano (fastest, recommended for Robotron)
- `yolo11s.pt` - Small
- `yolo11m.pt` - Medium
- `yolo11l.pt` - Large
- `yolo11x.pt` - Extra Large

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
