import os
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = False
Image.MAX_IMAGE_PIXELS = None

data_path = "./data/archive/train"
removed = 0
checked = 0

for class_name in os.listdir(data_path):
    class_dir = os.path.join(data_path, class_name)
    if not os.path.isdir(class_dir):
        continue
    for fname in os.listdir(class_dir):
        fpath = os.path.join(class_dir, fname)
        checked += 1
        try:
            with Image.open(fpath) as img:
                img.convert("RGB")  # full decode, catches truncation
        except Exception as e:
            print(f"Removing: {fpath} — {e}")
            os.remove(fpath)
            removed += 1

print(f"\nDone. Checked: {checked} | Removed: {removed} | Clean: {checked - removed}")
