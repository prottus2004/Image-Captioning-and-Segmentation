import streamlit as st
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as T
import torchvision
import random

# ---- CONFIG ----
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_CLASSES = 81  # for your U-Net
IMAGE_SIZE = (128, 128)

# ---- Load Models ----
@st.cache_resource
def load_unet():
    from Unet import UNetResNet
    model = UNetResNet(n_classes=N_CLASSES, out_size=IMAGE_SIZE)
    model.load_state_dict(torch.load("unet_coco_best1.pth", map_location=DEVICE))
    model.eval()
    model.to(DEVICE)
    return model

@st.cache_resource
def load_maskrcnn():
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    model.to(DEVICE)
    return model

unet = load_unet()
maskrcnn = load_maskrcnn()

# ---- COCO Class Names ----
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
    'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
    'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
    'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
    'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# ---- Streamlit UI ----
st.title("Semantic + Instance Segmentation Demo")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    # Load and preprocess image
    orig_image = Image.open(uploaded_file).convert("RGB")
    image_for_unet = orig_image.resize(IMAGE_SIZE)
    img_tensor_unet = T.ToTensor()(image_for_unet).unsqueeze(0).to(DEVICE)
    img_tensor_maskrcnn = T.ToTensor()(orig_image).unsqueeze(0).to(DEVICE)

    # Semantic Segmentation (U-Net)
    with torch.no_grad():
        output = unet(img_tensor_unet)
        pred_mask = torch.argmax(output, dim=1).squeeze().cpu().numpy().astype(np.uint8)

    # Colorize mask (random palette)
    def colorize_mask(mask):
        palette = np.random.randint(0, 255, (N_CLASSES, 3), dtype=np.uint8)
        return palette[mask]
    color_mask = colorize_mask(pred_mask)
    mask_img = Image.fromarray(color_mask).resize(orig_image.size)

    # Overlay for semantic
    overlay = np.array(orig_image).copy()
    alpha = 0.5
    overlay = (alpha * overlay + (1 - alpha) * np.array(mask_img)).astype(np.uint8)

    # Instance Segmentation (Mask R-CNN)
    with torch.no_grad():
        pred = maskrcnn(img_tensor_maskrcnn)[0]

    instance_image = orig_image.copy()
    draw = ImageDraw.Draw(instance_image)
    for i in range(len(pred['boxes'])):
        score = pred['scores'][i].item()
        if score < 0.5:
            continue
        box = pred['boxes'][i].cpu().numpy()
        label = COCO_INSTANCE_CATEGORY_NAMES[pred['labels'][i].item()]
        color = tuple([random.randint(0,255) for _ in range(3)])
        draw.rectangle(box.tolist(), outline=color, width=3)
        draw.text((box[0], box[1]), f"{label} {score:.2f}", fill=color)
        # Draw mask
        mask = pred['masks'][i, 0].cpu().numpy()
        mask_img_instance = Image.fromarray((mask > 0.5).astype(np.uint8)*128)
        mask_img_instance = mask_img_instance.resize(orig_image.size)
        instance_image.paste(mask_img_instance, (0,0), mask_img_instance)

    # Show results
    st.subheader("Original Image")
    st.image(orig_image, use_column_width=True)
    st.subheader("Semantic Segmentation (U-Net)")
    st.image(mask_img, use_column_width=True)
    st.subheader("Semantic Overlay")
    st.image(overlay, use_column_width=True)
    st.subheader("Instance Segmentation (Maskrcnn)")
    st.image(instance_image, use_column_width=True)
else:
    st.info("Upload an image to see segmentation results.")
