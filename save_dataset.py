import argparse
import json
import os
from pathlib import Path

import labelbox
import requests
import shutil
import yaml
from dotenv import load_dotenv, dotenv_values
from PIL import Image
from tqdm import tqdm

load_dotenv()


def make_dirs(dir='new_dir/', force=False):
    # Create folders (optionally delete existing)
    dir = Path(dir)
    if dir.exists() and force:
        shutil.rmtree(dir)  # delete dir only if force flag is set
    for p in dir, dir / 'labels', dir / 'images':
        p.mkdir(parents=True, exist_ok=True)  # make dir
    return dir


def main(api_key: str, project_id: str, dataset_id: str, save_dir: str = './datasets/', force: bool = False):
    lb = labelbox.Client(api_key=api_key)
    project = lb.get_project(project_id)
    dataset = lb.get_dataset(dataset_id)

    dir = make_dirs(save_dir + dataset.name, force=force)

    # Get the class names from labelbox ontology
    class_names = []
    for tool in project.ontology().tools():
        class_names.append(tool.name)

    downloaded = 0
    skipped = 0
    skipped_no_labels = 0

    for img in tqdm(dataset.data_rows(), desc='Processing images', total=dataset.row_count):
        for raw_labels in img.labels():
            labels = json.loads(raw_labels.label)
            if labels is None or 'objects' not in labels or len(labels['objects']) == 0:
                skipped_no_labels += 1
                continue
            objects = labels['objects']
            # Clean up the external_id: remove Windows-style path prefixes
            img_name = os.path.basename(img.external_id.replace('\\', '/'))

            img_url = img.row_data
            img_path = dir / 'images' / img_name
            label_path = dir / 'labels' / (os.path.splitext(img_name)[0] + '.txt')

            # Check if image already exists
            if img_path.exists() and not force:
                # Still need image dimensions for label conversion
                im = Image.open(img_path)
                width, height = im.size
                skipped += 1
            else:
                # Download image
                im = Image.open(requests.get(img_url, stream=True).raw if img_url.startswith('http') else img_url)
                width, height = im.size
                im.save(img_path, quality=95, subsampling=0)
                downloaded += 1

            # Always update labels (they may have changed in Labelbox)
            with open(label_path, 'w') as f:
                for obj in objects:
                    label = obj['title']
                    cls = class_names.index(label)
                    bbox = obj['bbox']
                    x, y, w, h = bbox['left'], bbox['top'], bbox['width'], bbox['height']
                    x_center = (x + w / 2) / width
                    y_center = (y + h / 2) / height
                    f.write(f'{cls} {x_center} {y_center} {w / width} {h / height}\n')

    print(f"\nSummary: {downloaded} downloaded, {skipped} skipped (already exist), {skipped_no_labels} skipped (no labels)")

    # Save dataset.yaml
    d = {'path': f"../{dir}",   # dataset root dir"
         'train': "images",
         'val': "images",
         'test': "",  # optional
         'nc': len(class_names),
         'names': class_names}  # dictionary
    with open(dir / 'dataset.yaml', 'w') as f:
        yaml.dump(d, f, sort_keys=False)

    # Save class names
    with open(dir / 'classes.txt', 'w') as f:
        f.write('\n'.join(class_names))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download Labelbox dataset to YOLO format')
    parser.add_argument('--force', '-f', action='store_true',
                        help='Force re-download all images (default: only download new images)')
    args = parser.parse_args()

    config = dotenv_values(".env")
    api_key = config['API_KEY']
    project_id = config['PROJECT_ID']
    dataset_id = config['DATASET_ID']
    main(api_key, project_id, dataset_id, './datasets/', force=args.force)
