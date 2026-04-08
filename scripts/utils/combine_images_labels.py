import os
import json
import shutil
from PIL import Image

class_map = {
    "head": 0, "face": 1, "hand": 2, "gloves": 3, "mask": 4, 
    "helmet": 5, "safety_goggles": 6, "sunglass_goggles": 7, 
    "vest": 8, "spectales_googles": 9, "safety_shoes": 10, "casual_shoes": 11
}


def image_yolo_files(data_dir, output_dir):
    
    os.makedirs(output_dir, exist_ok=True)
    
    for file in os.listdir(data_dir):

        if not file.endswith(".json"):
            continue

        json_path = os.path.join(data_dir, file)

        with open(json_path, "r") as f:
            data = json.load(f)

        base_name = os.path.splitext(file)[0]

        # find image
        image_path = None
        for ext in [".jpg", ".png", ".jpeg", ".JPG", ".PNG", ".JPEG"]:
            temp_path = os.path.join(data_dir, base_name + ext)
            if os.path.exists(temp_path):
                image_path = temp_path
                image_ext = ext
                break

        if image_path is None:
            print(f" Image not found for {file}")
            continue

        # load image size
        img = Image.open(image_path)
        img_w, img_h = img.size

        yolo_lines = []

        # convert annotations
        for shape in data.get("shapes", []):
            label = shape.get("label", "").strip().lower()
            label = label.replace(" ", "_").replace("-", "_")

            if label not in class_map:
                print(f"Skipping: {label}")
                continue

            points = shape.get("points", [])
            if not points:
                continue

            xs = [p[0] for p in points]
            ys = [p[1] for p in points]

            x_min = max(0, min(xs))
            x_max = min(img_w, max(xs))
            y_min = max(0, min(ys))
            y_max = min(img_h, max(ys))

            x_center = ((x_min + x_max) / 2) / img_w
            y_center = ((y_min + y_max) / 2) / img_h
            width = (x_max - x_min) / img_w
            height = (y_max - y_min) / img_h

            class_id = class_map[label]

            yolo_lines.append(
                f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
            )

        # copy image to new folder
        new_image_path = os.path.join(output_dir, base_name + image_ext)
        shutil.copy(image_path, new_image_path)

        # save txt in new folder
        txt_path = os.path.join(output_dir, base_name + ".txt")
        with open(txt_path, "w") as f:
            f.write("\n".join(yolo_lines))

        print(f"Done: {base_name}")


if __name__ == "__main__":

    data_dir = "data/raw/folder11"
    output_dir = "data/preprocess/yolo_dataset"