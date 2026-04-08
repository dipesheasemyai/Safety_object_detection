import os
import json
from PIL import Image

class_map = {
    "head": 0, "face": 1, "hand": 2, "gloves": 3, "mask": 4, 
    "helmet": 5, "safety_goggle": 6, "sunglass_goggle": 7, 
    "vest": 8, "spectales": 9, "safety_shoes": 10, "casual_shoes": 11
}

def json_to_txt_file(data_dir):
    output_dir = os.path.join(data_dir, "labels")
    os.makedirs(output_dir, exist_ok=True)

    for file in os.listdir(data_dir):
        if not file.endswith(".json"):
            continue

        json_path = os.path.join(data_dir, file)
        with open(json_path, "r") as f:
            data = json.load(f)

        base_name = os.path.splitext(file)[0]

        # Find corresponding image
        image_path = None
        for ext in [".jpg", ".png", ".jpeg", ".JPG", ".PNG", ".JPEG"]:
            temp_path = os.path.join(data_dir, base_name + ext)
            if os.path.exists(temp_path):
                image_path = temp_path
                break

        if image_path is None:
            print(f"Image not found for {file}")
            continue

        # Open image correctly
        img = Image.open(image_path)
        img_w, img_h = img.size

        yolo_lines = []

        for shape in data.get("shapes", []):
            label = shape.get("label", "").strip().lower()

            if label not in class_map:
                print(f"Skipping unknown label: {label}")
                continue

            points = shape.get("points", [])
            if not points:
                continue

            # Extract bbox correctly
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]

            x_min, x_max = max(0, min(xs)), min(img_w, max(xs))
            y_min, y_max = max(0, min(ys)), min(img_h, max(ys))

            # YOLO format
            x_center = ((x_min + x_max) / 2) / img_w
            y_center = ((y_min + y_max) / 2) / img_h
            width = (x_max - x_min) / img_w
            height = (y_max - y_min) / img_h

            class_id = class_map[label]
            line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
            yolo_lines.append(line)

        # Save labels
        out_path = os.path.join(output_dir, base_name + ".txt")
        with open(out_path, "w") as f:
            f.write("\n".join(yolo_lines))

        print(f"Converted: {file} -> {base_name}.txt")


if __name__ == "__main__":
    data_dir = "data/raw/folder11"
    json_to_txt_file(data_dir)