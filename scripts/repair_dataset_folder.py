import argparse
from extensions_built_in.dataset_tools.tools.image_tools import load_image
from tqdm import tqdm
import os

parser = argparse.ArgumentParser(description='Process some images.')
parser.add_argument("input_folder", type=str, help="Path to folder containing images")

args = parser.parse_args()

img_types = ['.jpg', '.jpeg', '.png', '.webp']

# find all images in the input folder
images = []
for root, _, files in os.walk(args.input_folder):
    for file in files:
        if file.lower().endswith(tuple(img_types)):
            images.append(os.path.join(root, file))
print(f"Found {len(images)} images")

num_skipped = 0
num_repaired = 0
num_deleted = 0

pbar = tqdm(total=len(images), desc=f"Repaired {num_repaired} images", unit="image")
for img_path in images:
    filename = os.path.basename(img_path)
    filename_no_ext, file_extension = os.path.splitext(filename)
    # if it is jpg, ignore
    if file_extension.lower() == '.jpg':
        num_skipped += 1
        pbar.update(1)

        continue

    try:
        img = load_image(img_path)
    except Exception as e:
        print(f"Error opening {img_path}: {e}")
        # delete it
        os.remove(img_path)
        num_deleted += 1
        pbar.update(1)
        pbar.set_description(f"Repaired {num_repaired} images, Skipped {num_skipped}, Deleted {num_deleted}")
        continue

    new_path = os.path.join(os.path.dirname(img_path), filename_no_ext + '.jpg')

    img = img.convert("RGB")
    img.save(new_path, quality=95)
    # remove the old file
    os.remove(img_path)
    num_repaired += 1
    pbar.update(1)
    # update pbar
    pbar.set_description(f"Repaired {num_repaired} images, Skipped {num_skipped}, Deleted {num_deleted}")

print("Done")