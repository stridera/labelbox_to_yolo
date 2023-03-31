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


def make_dirs(dir='new_dir/'):
    # Create folders
    dir = Path(dir)
    if dir.exists():
        shutil.rmtree(dir)  # delete dir
    for p in dir, dir / 'labels', dir / 'images':
        p.mkdir(parents=True, exist_ok=True)  # make dir
    return dir


def main(api_key: str, project_id: str, dataset_id: str, save_dir: str = './datasets/'):
    lb = labelbox.Client(api_key=api_key)
    project = lb.get_project(project_id)
    dataset = lb.get_dataset(dataset_id)

    dir = make_dirs(save_dir + dataset.name)

    # Get the class names from labelbox ontology
    class_names = []
    for tool in project.ontology().tools():
        class_names.append(tool.name)

    for img in tqdm(dataset.data_rows(), desc='Downloading images and labels', total=dataset.row_count):
        for raw_labels in img.labels():
            labels = json.loads(raw_labels.label)
            if labels is None or 'objects' not in labels or len(labels['objects']) == 0:
                continue
            objects = labels['objects']
            img_name = img.external_id

            img_url = img.row_data
            img_path = dir / 'images' / img_name
            label_path = dir / 'labels' / (os.path.splitext(img_name)[0] + '.txt')

            # Download image
            im = Image.open(requests.get(img_url, stream=True).raw if img_url.startswith('http') else img_url)  # open
            width, height = im.size  # image size
            im.save(img_path, quality=95, subsampling=0)

            # Save labels
            with open(label_path, 'w') as f:
                for obj in objects:
                    label = obj['title']
                    cls = class_names.index(label)
                    bbox = obj['bbox']
                    x, y, w, h = bbox['left'], bbox['top'], bbox['width'], bbox['height']
                    x_center = (x + w / 2) / width
                    y_center = (y + h / 2) / height
                    f.write(f'{cls} {x_center} {y_center} {w / width} {h / height}\n')

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
    config = dotenv_values(".env")
    api_key = config['API_KEY']
    project_id = config['PROJECT_ID']
    dataset_id = config['DATASET_ID']
    main(api_key, project_id, dataset_id, './datasets/')
