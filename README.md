# Labelbox to YOLOv5

## Introduction

Downloads a dataset from labelbox and formats it for yolov5.

## Installation

```
# Clone the repository
git clone --recurse-submodules git@github.com:stridera/robotron_classifier.git

# Create a virtual environment (optional)
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
# Install yolov5 dependencies to train
pip install -r yolov5/requirements.txt
```

### Create a .env file with the following variables

```
LABELBOX_API_KEY=<your labelbox api key>
PROJECT_ID=<your labelbox project id>
DATASET_NAME=<your labelbox dataset name>
```

## Usage

### Download and create the dataset

```
python ./save_dataset.py
```

### Train the model

```
cd yolov5
python train.py --img 640 --batch 16 --epochs 3 --data ../datasets/Robotron/dataset.yaml --weights yolov5s.pt
```

Note: It'll end with something like `Results saved to runs/train/exp15`. The `exp15` is the experiment name.

### Test the model

```
cd yolov5
python detect.py --data ../datasets/Robotron/dataset.yaml --weights runs/train/exp15/weights/best.pt --img 640 --conf 0.25 --source ../datasets/Robotron/images/
```

Note: Replace /exp/ with the name of your experiment
