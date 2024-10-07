# depth_estimation.py

import torch
import cv2

def estimate_depth(image_path):
    # Load MiDaS model
    model_type = "DPT_Large"  # Options: DPT_Large, DPT_Hybrid, MiDaS_small
    midas = torch.hub.load("intel-isl/MiDaS", model_type)

    # Load transforms to prepare input for the model
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

    if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
        transform = midas_transforms.dpt_transform
    else:
        transform = midas_transforms.small_transform

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    midas.to(device)
    midas.eval()

    # Load an image
    img = cv2.imread(image_path)

    # Convert OpenCV image (BGR) to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Prepare the image for inference
    input_batch = transform(img_rgb).to(device)

    # Perform depth estimation
    with torch.no_grad():
        prediction = midas(input_batch)

    # Resize the output to match the input size
    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=img_rgb.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze()

    # Convert to NumPy array
    depth_map = prediction.cpu().numpy()

    # Normalize the depth map
    depth_min = depth_map.min()
    depth_max = depth_map.max()
    depth_map = (depth_map - depth_min) / (depth_max - depth_min)

    return img_rgb, depth_map
