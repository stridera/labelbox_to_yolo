# YOLO11 Model Usage Guide

This guide explains how to use the trained YOLO11 model for Robotron sprite detection in your own Python code.

## Model Information

**Model:** YOLO11-nano (yolo11n)
**Classes:** 13 Robotron game sprites
**Input size:** 640x640
**Inference speed:** ~0.4ms per image (~2,500 FPS)
**Best model weights:** `runs/detect/train3/weights/best.pt`

### Performance Metrics (50 epochs)
- **mAP50:** 56.3%
- **mAP50-95:** 38.1%
- **Precision:** 75.8%
- **Recall:** 59.2%

### Detected Classes

| Class ID | Class Name | mAP50 | Recall | Notes |
|----------|------------|-------|--------|-------|
| 0 | Player | 91.8% | 64.2% | Excellent detection |
| 1 | Civilian | 76.6% | 66.8% | Excellent detection |
| 2 | Grunt | 71.5% | 82.5% | Excellent detection |
| 3 | Hulk | 69.3% | 57.0% | Good detection |
| 4 | Sphereoid | 74.1% | 69.5% | Good detection |
| 5 | Enforcer | 39.4% | 57.1% | Moderate detection |
| 6 | Brain | 58.5% | 75.7% | Good detection |
| 7 | Tank | 80.2% | 72.5% | Excellent detection |
| 8 | Quark | 59.0% | 48.8% | Moderate detection |
| 9 | Electrode | 65.5% | 72.1% | Good detection |
| 10 | Enforcer Bullet | 1.7% | 7.0% | Poor (very small) |
| 11 | Converted Civilian | 23.2% | 29.3% | Poor detection |
| 12 | Brain Bullet | 18.7% | 26.4% | Poor (very small) |

**Note:** Bullets have low detection rates due to their small size. Consider lowering confidence threshold for bullet detection or training longer.

## Installation

```bash
pip install ultralytics
```

Or if using Poetry:
```bash
poetry add ultralytics
```

## Basic Usage

### 1. Loading the Model

```python
from ultralytics import YOLO

# Load the trained model
model = YOLO('runs/detect/train3/weights/best.pt')

# Optional: Print model information
print(f"Model: {model.model_name}")
print(f"Classes: {model.names}")
```

### 2. Running Inference on a Single Image

```python
from ultralytics import YOLO
import cv2

# Load model
model = YOLO('runs/detect/train3/weights/best.pt')

# Run inference
results = model('path/to/robotron/screenshot.png', conf=0.25)

# Process results
for result in results:
    # Get bounding boxes
    boxes = result.boxes

    for box in boxes:
        # Get box coordinates (xyxy format)
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

        # Get confidence and class
        confidence = box.conf[0].cpu().numpy()
        class_id = int(box.cls[0].cpu().numpy())
        class_name = model.names[class_id]

        print(f"Detected {class_name} at ({x1}, {y1}, {x2}, {y2}) with confidence {confidence:.2f}")
```

### 3. Running Inference on Video/Stream (Real-time Gameplay)

```python
from ultralytics import YOLO
import cv2

# Load model
model = YOLO('runs/detect/train3/weights/best.pt')

# Open video stream (or capture from game window)
cap = cv2.VideoCapture('path/to/robotron/gameplay.mp4')

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Run inference on the frame
    results = model(frame, conf=0.25, verbose=False)

    # Visualize results on frame
    annotated_frame = results[0].plot()

    # Display the frame
    cv2.imshow('Robotron Detection', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### 4. Extracting Detection Data for Game AI

```python
from ultralytics import YOLO
import numpy as np

class RobotronDetector:
    def __init__(self, model_path='runs/detect/train3/weights/best.pt', conf_threshold=0.25):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold

    def detect_sprites(self, frame):
        """
        Detect all sprites in a frame and return structured data.

        Args:
            frame: numpy array (image)

        Returns:
            dict: Dictionary with sprite types as keys and lists of detections as values
        """
        results = self.model(frame, conf=self.conf_threshold, verbose=False)[0]

        detections = {
            'player': [],
            'enemies': [],
            'civilians': [],
            'bullets': []
        }

        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            confidence = float(box.conf[0].cpu().numpy())
            class_id = int(box.cls[0].cpu().numpy())
            class_name = self.model.names[class_id]

            # Calculate center point
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            detection = {
                'class_name': class_name,
                'class_id': class_id,
                'bbox': [x1, y1, x2, y2],
                'center': [center_x, center_y],
                'confidence': confidence
            }

            # Categorize detections
            if class_name == 'Player':
                detections['player'].append(detection)
            elif class_name == 'Civilian' or class_name == 'Converted Civilian':
                detections['civilians'].append(detection)
            elif 'Bullet' in class_name:
                detections['bullets'].append(detection)
            else:
                detections['enemies'].append(detection)

        return detections

    def get_nearest_threat(self, frame, player_position):
        """
        Find the nearest enemy to the player.

        Args:
            frame: numpy array (image)
            player_position: tuple (x, y) of player position

        Returns:
            dict: Nearest enemy detection or None
        """
        detections = self.detect_sprites(frame)

        if not detections['enemies']:
            return None

        nearest = None
        min_distance = float('inf')

        for enemy in detections['enemies']:
            enemy_pos = enemy['center']
            distance = np.sqrt(
                (enemy_pos[0] - player_position[0])**2 +
                (enemy_pos[1] - player_position[1])**2
            )

            if distance < min_distance:
                min_distance = distance
                nearest = enemy
                nearest['distance'] = distance

        return nearest

# Example usage
detector = RobotronDetector()

# Process a frame
frame = cv2.imread('robotron_frame.png')
detections = detector.detect_sprites(frame)

print(f"Found {len(detections['player'])} players")
print(f"Found {len(detections['enemies'])} enemies")
print(f"Found {len(detections['civilians'])} civilians")
print(f"Found {len(detections['bullets'])} bullets")

# Get nearest threat
if detections['player']:
    player_pos = detections['player'][0]['center']
    threat = detector.get_nearest_threat(frame, player_pos)
    if threat:
        print(f"Nearest threat: {threat['class_name']} at distance {threat['distance']:.1f}")
```

### 5. Batch Processing Multiple Images

```python
from ultralytics import YOLO
from pathlib import Path

# Load model
model = YOLO('runs/detect/train3/weights/best.pt')

# Process all images in a directory
image_dir = Path('datasets/Robotron/images')
results = model(list(image_dir.glob('*.png')), conf=0.25)

# Save annotated images
for i, result in enumerate(results):
    result.save(filename=f'output/result_{i}.png')
```

## Advanced Configuration

### Adjusting Confidence Threshold

```python
# Lower threshold for better recall (more detections, more false positives)
results = model(frame, conf=0.15)

# Higher threshold for better precision (fewer detections, fewer false positives)
results = model(frame, conf=0.50)
```

### Filtering by Class

```python
# Detect only specific classes (e.g., only enemies)
enemy_classes = [2, 3, 4, 5, 6, 7, 8, 9]  # Grunt, Hulk, Sphereoid, etc.
results = model(frame, classes=enemy_classes, conf=0.25)
```

### Adjusting IoU Threshold (Non-Maximum Suppression)

```python
# Lower IoU = more aggressive suppression of overlapping boxes
results = model(frame, conf=0.25, iou=0.45)

# Higher IoU = keep more overlapping boxes (useful for crowded scenes)
results = model(frame, conf=0.25, iou=0.70)
```

### Device Selection

```python
# Use CPU
model = YOLO('runs/detect/train3/weights/best.pt')
results = model(frame, device='cpu')

# Use GPU
results = model(frame, device='cuda:0')  # or just 'cuda'

# Use MPS (Mac M1/M2)
results = model(frame, device='mps')
```

## Performance Optimization

### 1. Use Half Precision (FP16) for Faster Inference

```python
# Requires CUDA GPU
results = model(frame, half=True)
```

### 2. Batch Processing for Multiple Frames

```python
import numpy as np

# Stack multiple frames
frames = [frame1, frame2, frame3, frame4]
results = model(frames, conf=0.25)
```

### 3. Image Size Optimization

```python
# Smaller image = faster inference, lower accuracy
results = model(frame, imgsz=320)  # default is 640

# Larger image = slower inference, higher accuracy
results = model(frame, imgsz=1280)
```

## Integration Examples

### Game AI Agent

```python
class RobotronAI:
    def __init__(self):
        self.detector = RobotronDetector()

    def decide_action(self, frame):
        """Decide next action based on detections."""
        detections = self.detector.detect_sprites(frame)

        if not detections['player']:
            return 'wait'

        player = detections['player'][0]
        player_pos = player['center']

        # Priority 1: Save civilians
        if detections['civilians']:
            nearest_civilian = min(
                detections['civilians'],
                key=lambda c: self.distance(player_pos, c['center'])
            )
            return self.move_towards(player_pos, nearest_civilian['center'])

        # Priority 2: Avoid bullets
        dangerous_bullets = [
            b for b in detections['bullets']
            if self.distance(player_pos, b['center']) < 100
        ]
        if dangerous_bullets:
            return self.evade(player_pos, dangerous_bullets)

        # Priority 3: Attack nearest enemy
        if detections['enemies']:
            nearest_enemy = min(
                detections['enemies'],
                key=lambda e: self.distance(player_pos, e['center'])
            )
            return self.move_towards(player_pos, nearest_enemy['center'])

        return 'patrol'

    def distance(self, pos1, pos2):
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

    def move_towards(self, current, target):
        # Return direction to move
        dx = target[0] - current[0]
        dy = target[1] - current[1]
        return {'action': 'move', 'direction': (dx, dy)}

    def evade(self, current, bullets):
        # Calculate safe direction away from bullets
        # Implementation here...
        return {'action': 'evade'}
```

## Troubleshooting

### Low Detection Accuracy
- Lower confidence threshold: `conf=0.15`
- Check if image size matches training: `imgsz=640`
- Ensure good image quality and lighting

### Too Many False Positives
- Raise confidence threshold: `conf=0.35` or higher
- Adjust IoU threshold: `iou=0.45`

### Slow Inference
- Use smaller image size: `imgsz=320`
- Enable half precision: `half=True` (GPU only)
- Ensure GPU is being used: `device='cuda'`

### Missing Small Objects (Bullets)
- Increase image size: `imgsz=1280`
- Lower confidence threshold for bullet classes
- Consider retraining with more bullet examples

## Model Export for Deployment

### Export to ONNX (Cross-platform)

```python
from ultralytics import YOLO

model = YOLO('runs/detect/train3/weights/best.pt')
model.export(format='onnx', imgsz=640)
```

### Export to TensorRT (NVIDIA GPUs)

```python
model.export(format='engine', imgsz=640, half=True)
```

### Export to CoreML (Apple devices)

```python
model.export(format='coreml', imgsz=640)
```

## Further Reading

- [Ultralytics YOLO11 Documentation](https://docs.ultralytics.com/models/yolo11/)
- [Python API Reference](https://docs.ultralytics.com/usage/python/)
- [Model Export Guide](https://docs.ultralytics.com/modes/export/)
