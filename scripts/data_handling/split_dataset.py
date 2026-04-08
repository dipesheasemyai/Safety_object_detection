import os
import shutil
import random


source_dir = "data/preprocess/yolo_dataset"

output_dir = "/home/easemyai/Documents/task/project/dataset"

split_ratio = 0.8  

for path in [
    "images/train", "images/val",
    "labels/train", "labels/val"
]:
    os.makedirs(os.path.join(output_dir, path), exist_ok=True)


image_files = []
for file in os.listdir(source_dir):
    if file.lower().endswith((".jpg", ".png", ".jpeg")):
        image_files.append(file)

random.shuffle(image_files)

split_index = int(len(image_files) * split_ratio)
train_files = image_files[:split_index]
val_files = image_files[split_index:]

def move_files(file_list, split_type):
    for img_file in file_list:
        base_name = os.path.splitext(img_file)[0]
        txt_file = base_name + ".txt"

        src_img = os.path.join(source_dir, img_file)
        src_txt = os.path.join(source_dir, txt_file)

        dst_img = os.path.join(output_dir, "images", split_type, img_file)
        dst_txt = os.path.join(output_dir, "labels", split_type, txt_file)

        shutil.copy(src_img, dst_img)

        if os.path.exists(src_txt):
            shutil.copy(src_txt, dst_txt)
        else:
            print(f"Missing label: {txt_file}")

move_files(train_files, "train")
move_files(val_files, "val")

print(f"Train: {len(train_files)} images")
print(f"Val: {len(val_files)} images")