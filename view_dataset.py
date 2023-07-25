"""
Display images with their corresponding labels
"""

import os
import argparse
import matplotlib.pyplot as plt
from PIL import Image


# def show_image(img_path: str, label_path: str, class_names: list):
#     """ Show the image with the labels """
#     # Read the image
#     img = Image.open(img_path)
#     width, height = img.size

#     # Read the labels
#     with open(label_path) as f:
#         for line in f:
#             cls, x, y, w, h = line.strip().split()
#             cls = int(cls)
#             x, y, w, h = float(x), float(y), float(w), float(h)
#             x1, y1 = x - w / 2, y - h / 2
#             x2, y2 = x + w / 2, y + h / 2

#             # Draw the bounding box
#             plt.figure()
#             plt.imshow(img)
#             plt.plot([x1 * width, x2 * width, x2 * width, x1 * width, x1 * width],
#                      [y1 * height, y1 * height, y2 * height, y2 * height, y1 * height])
#             plt.title(class_names[cls])
#             plt.axis('off')
#             plt.tight_layout()
#             plt.show()
    
    


def main(dataset_path: str):
    # Get the class names from labelbox ontology
    class_names = []
    with open(os.path.join(dataset_path, 'classes.txt')) as f:
        for line in f:
            class_names.append(line.strip())

    # Get the image and label paths
    img_paths = []
    label_paths = []
    for img_name in os.listdir(os.path.join(dataset_path, 'images')):
        img_paths.append(os.path.join(dataset_path, 'images', img_name))
        label_paths.append(os.path.join(dataset_path, 'labels', os.path.splitext(img_name)[0] + '.txt'))
    
    # Show a window and let us scroll through the images
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.2)
    ax.set_title('Scroll to navigate images')

    # Add the slider
    ax_slider = plt.axes((0.1, 0.1, 0.8, 0.05))
    slider = plt.Slider(ax_slider, 'Image', 0, len(img_paths) - 1, valinit=0, valstep=1)

    # Update the image when the slider is changed
    def update(val):
        print("Updating image", val)
        img = Image.open(img_paths[val])
        
        # Draw the images with labeled rectangles
        ax.clear()
        ax.imshow(img)

        # Read the labels
        with open(label_paths[val]) as f:
            for line in f:
                cls, x, y, w, h = line.strip().split()
                cls = int(cls)
                x, y, w, h = float(x), float(y), float(w), float(h)
                x1, y1 = x - w / 2, y - h / 2
                x2, y2 = x + w / 2, y + h / 2

                # Draw the bounding box
                ax.plot([x1 * img.width, x2 * img.width, x2 * img.width, x1 * img.width, x1 * img.width],
                        [y1 * img.height, y1 * img.height, y2 * img.height, y2 * img.height, y1 * img.height])
                ax.text(x1 * img.width, y1 * img.height, class_names[cls], fontsize=10, bbox=dict(facecolor='yellow', alpha=0.5))

        ax.axis('off')
        ax.set_title(f'{img_paths[val]}')
        fig.canvas.draw_idle()

    slider.on_changed(update)
    
    # Make arrows move slider
    def on_key(event):
        if event.key == 'left':
            slider.val = max(slider.val - 1, 0)
        elif event.key == 'right':
            slider.val = min(slider.val + 1, len(img_paths) - 1)
        slider.set_val(slider.val)
        # update(slider.val)
        fig.canvas.draw_idle()
    fig.canvas.mpl_connect('key_press_event', on_key)
    
    update(0)
    plt.show()



if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='./datasets')
    args = parser.parse_args()
    main(args.dataset)