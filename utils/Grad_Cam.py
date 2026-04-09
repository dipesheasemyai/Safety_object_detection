import cv2
import numpy as np
import torch
import torch.nn as nn
from ultralytics import YOLO
from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


def Grad_Cam_process(model_path, image_path):

    class YOLOWrapper(nn.Module):
        def __init__(self, yolo_model):
            super().__init__()
            self.model = yolo_model

        def forward(self, x):
            outputs = self.model(x)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            return outputs


    # Load model
    yolo = YOLO(model_path)
    yolo_model = yolo.model
    wrapped_model = YOLOWrapper(yolo_model)

    # MULTIPLE LAYERS (better representation)
    target_layers = [
        yolo_model.model[-1],
        yolo_model.model[-2],
        yolo_model.model[-3]
    ]

    # Load image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img_norm = img.astype(np.float32) / 255.0

    input_tensor = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0)

    # EigenCAM (PCA-based)
    cam = EigenCAM(
        model=wrapped_model,
        target_layers=target_layers
    )

    grayscale_cam = cam(input_tensor=input_tensor)[0]

    visualization = show_cam_on_image(img_norm, grayscale_cam, use_rgb=True)

    cv2.imwrite("final_output7.jpg", cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    model_path = "/home/easemyai/Documents/mypycodes/dipesh_folder/face_detection_models/detection_model.pt"
    image_path = "/home/easemyai/Documents/mypycodes/dipesh_folder/face_detection_models/unnamed.jpg"
    Grad_Cam_process(model_path, image_path)