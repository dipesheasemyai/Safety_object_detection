import cv2
import numpy as np
import torch
import torch.nn as nn
from ultralytics import YOLO
from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


def heatmap_boundingbox_pretrained_model(model_path, image_path):
    class YOLOWrapper(nn.Module):
        def __init__(self, yolo_model):
            super().__init__()
            self.model = yolo_model

        def forward(self, x):
            outputs = self.model(x)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            return outputs

    class YOLOWrapper(nn.Module):
        def __init__(self, yolo_model):
            super().__init__()
            self.model = yolo_model

        def forward(self, x):
            outputs = self.model(x)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            return outputs


    yolo = YOLO(model_path)
    yolo_model = yolo.model
    wrapped_model = YOLOWrapper(yolo_model)

    target_layers = [yolo_model.model[-2]]


    results = yolo(image_path)[0]


    boxed_img = results.plot()  # already has boxes

    boxed_img = cv2.cvtColor(boxed_img, cv2.COLOR_BGR2RGB)


    img_norm = boxed_img.astype(np.float32) / 255.0

    input_tensor = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0)


    cam = EigenCAM(model=wrapped_model, target_layers=target_layers)
    grayscale_cam = cam(input_tensor=input_tensor)[0]

    # Overlay CAM
    visualization = show_cam_on_image(img_norm, grayscale_cam, use_rgb=True)

    # Save
    cv2.imwrite("output/final_output.jpg", cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))

    yolo = YOLO(model_path)
    yolo_model = yolo.model
    wrapped_model = YOLOWrapper(yolo_model)

    target_layers = [yolo_model.model[-2]]


    results = yolo(image_path)[0]


    boxed_img = results.plot()  # already has boxes

    boxed_img = cv2.cvtColor(boxed_img, cv2.COLOR_BGR2RGB)


    img_norm = boxed_img.astype(np.float32) / 255.0

    input_tensor = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0)


    cam = EigenCAM(model=wrapped_model, target_layers=target_layers)
    grayscale_cam = cam(input_tensor=input_tensor)[0]

    # Overlay CAM
    visualization = show_cam_on_image(img_norm, grayscale_cam, use_rgb=True)

    # Save
    cv2.imwrite("output/final_output1.jpg", cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))

if __name__ == "__main__":
    model_path = "/home/easemyai/dipesh_folder/models/yolov8n.pt"
    image_path = "/home/easemyai/dipesh_folder/task/folder26/construction_site_construction_workers_work_workers_building_job_project-599609_jpg.rf.3bc7e249a0772f88b452af2b842bf541.jpg"
    heatmap_boundingbox_pretrained_model(model_path, image_path)
