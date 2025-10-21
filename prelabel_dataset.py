import argparse
from pathlib import Path
from typing import List, Dict

import labelbox
from labelbox.data.annotation_types import Label, ObjectAnnotation, Rectangle, Point, GenericDataRowData
import requests
from dotenv import load_dotenv, dotenv_values
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO

load_dotenv()


def download_image(url: str, save_path: Path) -> Image.Image:
    """Download an image from a URL and save it locally."""
    response = requests.get(url, stream=True)
    response.raise_for_status()
    im = Image.open(response.raw)
    im.save(save_path, quality=95, subsampling=0)
    return im


def run_inference(model: YOLO, image_path: Path, conf_threshold: float = 0.3) -> List[Dict]:
    """
    Run YOLO inference on an image and return predictions above confidence threshold.

    Returns:
        List of dicts with keys: class_id, class_name, confidence, bbox (in YOLO format)
    """
    results = model.predict(source=str(image_path), conf=conf_threshold, verbose=False)

    predictions = []
    for result in results:
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            continue

        for box in boxes:
            # Get box coordinates in xyxy format
            xyxy = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0].cpu().numpy())
            cls_id = int(box.cls[0].cpu().numpy())
            cls_name = model.names[cls_id]

            # Convert to xywh (center format, normalized)
            xywhn = box.xywhn[0].cpu().numpy()

            predictions.append({
                'class_id': cls_id,
                'class_name': cls_name,
                'confidence': conf,
                'bbox_yolo': xywhn.tolist(),  # [x_center, y_center, width, height] normalized
                'bbox_xyxy': xyxy.tolist()  # [x1, y1, x2, y2] absolute pixels
            })

    return predictions


def yolo_to_labelbox_bbox(yolo_bbox: List[float], img_width: int, img_height: int) -> Dict:
    """
    Convert YOLO format bbox to Labelbox format.

    Args:
        yolo_bbox: [x_center, y_center, width, height] all normalized (0-1)
        img_width: Image width in pixels
        img_height: Image height in pixels

    Returns:
        Dict with keys: top, left, height, width (all in absolute pixels)
    """
    x_center, y_center, width, height = yolo_bbox

    # Convert to absolute pixels
    abs_width = width * img_width
    abs_height = height * img_height
    abs_x_center = x_center * img_width
    abs_y_center = y_center * img_height

    # Convert to top-left corner
    left = abs_x_center - (abs_width / 2)
    top = abs_y_center - (abs_height / 2)

    return {
        'top': int(top),
        'left': int(left),
        'height': int(abs_height),
        'width': int(abs_width)
    }


def create_label_annotation(data_row_id: str, external_id: str, predictions: List[Dict],
                           img_width: int, img_height: int,
                           ontology_index: Dict[str, str]) -> Label:
    """
    Create a Label with ObjectAnnotations for Labelbox upload using Python Annotation Types.

    Args:
        data_row_id: Labelbox data row ID
        predictions: List of prediction dicts from run_inference
        img_width: Image width in pixels
        img_height: Image height in pixels
        ontology_index: Mapping from class_name to feature_schema_id

    Returns:
        Label object with bounding box annotations
    """
    annotations = []

    for pred in predictions:
        class_name = pred['class_name']

        # Skip if class not in ontology
        if class_name not in ontology_index:
            print(f"Warning: Class '{class_name}' not found in project ontology, skipping")
            continue

        lb_bbox = yolo_to_labelbox_bbox(pred['bbox_yolo'], img_width, img_height)

        # Create ObjectAnnotation using Python types
        # Rectangle uses start (top-left) and end (bottom-right) points
        bbox_annotation = ObjectAnnotation(
            name=class_name,
            confidence=pred['confidence'],
            value=Rectangle(
                start=Point(x=lb_bbox['left'], y=lb_bbox['top']),
                end=Point(
                    x=lb_bbox['left'] + lb_bbox['width'],
                    y=lb_bbox['top'] + lb_bbox['height']
                )
            ),
            feature_schema_id=ontology_index[class_name]
        )
        annotations.append(bbox_annotation)

    # Create a Label containing all annotations for this data row
    # Use external_id (global_key) to reference the data row
    label = Label(
        data={"global_key": external_id},
        annotations=annotations
    )

    return label


def get_ontology_index(project) -> Dict[str, str]:
    """
    Get a mapping of class names to feature schema IDs from the project ontology.

    Returns:
        Dict mapping class_name -> feature_schema_id
    """
    ontology_index = {}
    for tool in project.ontology().tools():
        ontology_index[tool.name] = tool.feature_schema_id
    return ontology_index


def main(api_key: str, project_id: str, model_path: str,
         conf_threshold: float = 0.3, temp_dir: str = './temp_prelabel/',
         batch_size: int = 100):
    """
    Main pre-labeling workflow:
    1. Connect to Labelbox and get unlabeled data rows
    2. Download images
    3. Run YOLO inference
    4. Convert predictions to Labelbox format
    5. Upload as pre-labels

    Args:
        api_key: Labelbox API key
        project_id: Labelbox project ID
        model_path: Path to trained YOLO model weights (e.g., runs/detect/train/weights/best.pt)
        conf_threshold: Confidence threshold for predictions (default 0.3)
        temp_dir: Temporary directory for downloading images
        batch_size: Number of annotations to upload in each batch
    """
    # Initialize Labelbox client
    lb = labelbox.Client(api_key=api_key)
    project = lb.get_project(project_id)

    print(f"Connected to project: {project.name}")

    # Get ontology mapping
    print("Fetching project ontology...")
    ontology_index = get_ontology_index(project)
    print(f"Found {len(ontology_index)} classes: {list(ontology_index.keys())}")

    # Load YOLO model
    print(f"Loading YOLO model from {model_path}...")
    model = YOLO(model_path)
    print(f"Model loaded. Classes: {model.names}")

    # Create temp directory
    temp_path = Path(temp_dir)
    temp_path.mkdir(parents=True, exist_ok=True)

    # Get queued (unlabeled) data rows from project batches
    print("Fetching data rows from project batches...")
    try:
        # Use export to get data rows without labels
        # Filter for data rows that haven't been labeled yet
        export_task = project.export(
            params={
                "performance_details": False,
                "label_details": True
            }
        )
        export_task.wait_till_done()

        # Parse the export results using buffered stream
        queued_data_rows = []

        # Use get_buffered_stream for efficient streaming
        for buffered_item in export_task.get_buffered_stream():
            item = buffered_item.json

            # Check if data row has any submitted labels
            has_labels = False
            if 'projects' in item:
                for proj_data in item['projects'].values():
                    if 'labels' in proj_data and len(proj_data['labels']) > 0:
                        has_labels = True
                        break

            if not has_labels:
                queued_data_rows.append(item)

        print(f"Found {len(queued_data_rows)} unlabeled data rows")

    except Exception as e:
        print(f"Error fetching data rows: {e}")
        import traceback
        traceback.print_exc()
        print("\nPlease ensure your project has data rows and the API key has correct permissions.")
        return

    if len(queued_data_rows) == 0:
        print("No unlabeled data rows found. Exiting.")
        return

    # Process each data row
    labels = []

    for data_row_dict in tqdm(queued_data_rows, desc="Processing images"):
        try:
            # Handle export format
            data_row_id = data_row_dict.get('data_row', {}).get('id')
            external_id = data_row_dict.get('data_row', {}).get('external_id', 'unknown')
            row_data = data_row_dict.get('data_row', {}).get('row_data')

            if not row_data:
                print(f"Warning: No image URL for data row {data_row_id}, skipping")
                continue

            # Download image
            img_filename = f"{external_id}"
            img_path = temp_path / img_filename

            try:
                im = download_image(row_data, img_path)
                img_width, img_height = im.size
            except Exception as e:
                print(f"Error downloading image {external_id}: {e}, skipping")
                continue

            # Run inference
            predictions = run_inference(model, img_path, conf_threshold)

            if len(predictions) == 0:
                # No predictions above threshold, skip this image
                continue

            # Convert to Labelbox Label using Python Annotation Types
            label = create_label_annotation(
                data_row_id, external_id, predictions, img_width, img_height, ontology_index
            )

            labels.append(label)

            # Clean up temp image
            img_path.unlink(missing_ok=True)

        except Exception as e:
            print(f"Error processing data row: {e}, skipping")
            continue

    print(f"\nGenerated {len(labels)} pre-label annotations")

    if len(labels) == 0:
        print("No annotations to upload. Exiting.")
        return

    # Upload annotations to Labelbox using MAL API
    print("Uploading pre-labels to Labelbox...")
    try:
        from labelbox import MALPredictionImport
        import uuid

        # Debug: check the first label structure
        print(f"Uploading {len(labels)} labels...")
        if labels:
            print(f"First label has {len(labels[0].annotations)} annotations")

        # Use create_from_objects to upload Label objects
        upload_job = MALPredictionImport.create_from_objects(
            client=lb,
            project_id=project_id,
            name=f'prelabels-{uuid.uuid4().hex[:8]}',
            predictions=labels
        )

        print(f"Upload job created: {upload_job.name}")
        print("Waiting for upload to complete...")
        upload_job.wait_till_done()

        print(f"Upload status: {upload_job.state}")

        # Check for errors (state can be enum or string)
        state_str = str(upload_job.state)
        if "FAILED" in state_str:
            print(f"\nUpload failed!")

            # Try to get detailed errors
            print(f"\nAttempting to fetch error details...")
            try:
                # Try to access errors - this may raise ValueError if error_file_url is None
                print("Checking errors property...")
                try:
                    errors_list = list(upload_job.errors)
                    if errors_list:
                        print(f"Found {len(errors_list)} errors:")
                        for i, error in enumerate(errors_list[:10]):  # Show first 10
                            print(f"  Error {i+1}: {error}")
                    else:
                        print("  No errors found in errors property")
                except ValueError as ve:
                    print(f"  Cannot access errors: {ve}")
                except Exception as ee:
                    print(f"  Error accessing errors property: {ee}")

                # Try to access statuses
                print("\nChecking statuses property...")
                try:
                    statuses_list = list(upload_job.statuses)
                    if statuses_list:
                        print(f"Found {len(statuses_list)} statuses:")
                        for i, status in enumerate(statuses_list[:10]):  # Show first 10
                            print(f"  Status {i+1}: {status}")
                    else:
                        print("  No statuses found")
                except Exception as se:
                    print(f"  Error accessing statuses property: {se}")

            except Exception as e:
                print(f"Unexpected error while fetching details: {e}")

            # Get available attributes for debugging
            print(f"\nAvailable upload job attributes:")
            print(f"  - error_file_url: {getattr(upload_job, 'error_file_url', 'N/A')}")
            print(f"  - status_file_url: {getattr(upload_job, 'status_file_url', 'N/A')}")
            print(f"  - input_file_url: {getattr(upload_job, 'input_file_url', 'N/A')}")

        elif "FINISHED" in state_str:
            print(f"Successfully uploaded {len(labels)} pre-label annotations!")
        else:
            print(f"Upload in unexpected state: {upload_job.state}")

    except Exception as e:
        print(f"Error uploading annotations: {e}")
        import traceback
        traceback.print_exc()
        print("\nUpload failed. Check the error above for details.")

    # Clean up temp directory
    print("\nCleaning up temporary files...")
    for file in temp_path.glob('*'):
        file.unlink(missing_ok=True)
    temp_path.rmdir()

    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Pre-label unlabeled Labelbox images using a trained YOLO model'
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to trained YOLO model weights (e.g., runs/detect/train/weights/best.pt)'
    )
    parser.add_argument(
        '--conf',
        type=float,
        default=0.3,
        help='Confidence threshold for predictions (default: 0.3)'
    )
    parser.add_argument(
        '--temp-dir',
        type=str,
        default='./temp_prelabel/',
        help='Temporary directory for downloading images (default: ./temp_prelabel/)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=100,
        help='Number of annotations to upload in each batch (default: 100)'
    )

    args = parser.parse_args()

    # Load credentials from .env
    config = dotenv_values(".env")
    api_key = config.get('API_KEY')
    project_id = config.get('PROJECT_ID')

    if not api_key or not project_id:
        print("Error: API_KEY and PROJECT_ID must be set in .env file")
        exit(1)

    main(
        api_key=api_key,
        project_id=project_id,
        model_path=args.model,
        conf_threshold=args.conf,
        temp_dir=args.temp_dir,
        batch_size=args.batch_size
    )
