import streamlit as st
import requests
import torch
import numpy as np
from PIL import Image, ImageDraw, UnidentifiedImageError
import torchvision.transforms as T
import torchvision
import tempfile
import cv2
import time
import base64
from io import BytesIO
import json
import logging
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
import random
import threading
import queue
from threading import Thread

# ---- PAGE CONFIG ----
st.set_page_config(
    page_title="Vision AI Suite - Professional Computer Vision Platform",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---- CONFIG ----
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_CLASSES = 81  # for your U-Net
IMAGE_SIZE = (128, 128)

# COCO class names for object detection
COCO_CLASSES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# ---- MODEL LOADING ----

# ---- Load U-Net Model ----
@st.cache_resource
def load_unet():
    """Loads the U-Net model for semantic segmentation."""
    try:
        from Unet import UNetResNet
        model = UNetResNet(n_classes=N_CLASSES, out_size=IMAGE_SIZE)
        # NOTE: Ensure 'unet_coco_best1.pth' is in your project's root directory.
        model.load_state_dict(torch.load("unet_coco_best1.pth", map_location=DEVICE))
        model.eval()
        model.to(DEVICE)
        return model
    except ImportError:
        st.warning("⚠️ U-Net model definition not found. Please ensure `Unet.py` is in the project directory. Segmentation feature will be disabled.")
        return None
    except FileNotFoundError:
        st.warning("⚠️ U-Net weights file ('unet_coco_best1.pth') not found. Segmentation feature will be disabled.")
        return None
    except Exception as e:
        st.error(f"❌ Error loading U-Net model: {e}")
        return None

# ---- Load Mask R-CNN Model ----
@st.cache_resource
def load_maskrcnn():
    """Loads the Mask R-CNN model for instance segmentation."""
    try:
        # Load pre-trained Mask R-CNN model
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights='DEFAULT')
        model.eval()
        model.to(DEVICE)
        # Removed success message
        return model
    except Exception as e:
        st.error(f"❌ Error loading Mask R-CNN model: {e}")
        return None

# ---- Load Object Detection Model ----
@st.cache_resource
def load_object_detection_model():
    """Loads the object detection model (Faster R-CNN)."""
    try:
        # Load pre-trained Faster R-CNN model
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='DEFAULT')
        model.eval()
        model.to(DEVICE)
        # Removed success message
        return model
    except Exception as e:
        st.error(f"❌ Error loading object detection model: {e}")
        return None

# ---- Load Caption Generation Model ----
try:
    # NOTE: Ensure 'inference.py' and its dependencies are in the project directory.
    from inference import generate_caption
except ImportError:
    st.warning("⚠️ Caption generation module not found. Please ensure `inference.py` is in the project directory. Captioning will be disabled.")
    def generate_caption(image_path):
        # Simple caption generation as fallback
        captions = [
            "A beautiful scene captured with intricate details and vibrant colors.",
            "An image showcasing various objects and elements in a natural setting.",
            "A photograph displaying rich textures and interesting visual composition.",
            "A scene with multiple elements creating a harmonious visual narrative.",
            "An image featuring diverse objects and subjects in their environment.",
            "A captivating photograph with excellent lighting and composition.",
            "A detailed image showing various subjects in their natural context.",
        ]
        return random.choice(captions)

# Initialize models on startup (silently)
@st.cache_resource
def initialize_models():
    """Initialize all models silently."""
    unet = load_unet()
    maskrcnn = load_maskrcnn()
    object_detection_model = load_object_detection_model()
    return unet, maskrcnn, object_detection_model

# Load models silently
unet, maskrcnn, object_detection_model = initialize_models()

# ---- IMAGE PROCESSING FUNCTIONS ----

def perform_edge_detection(image):
    """Perform various edge detection algorithms."""
    # Convert PIL to OpenCV format
    cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    
    # Different edge detection methods
    edges_canny = cv2.Canny(gray, 100, 200)
    edges_sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    edges_sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    edges_sobel = np.sqrt(edges_sobel_x**2 + edges_sobel_y**2)
    edges_laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    
    # Normalize and convert back to PIL
    edges_canny_pil = Image.fromarray(edges_canny)
    edges_sobel_pil = Image.fromarray(np.uint8(np.clip(edges_sobel, 0, 255)))
    edges_laplacian_pil = Image.fromarray(np.uint8(np.clip(np.abs(edges_laplacian), 0, 255)))
    
    return {
        'canny': edges_canny_pil,
        'sobel': edges_sobel_pil,
        'laplacian': edges_laplacian_pil
    }

def perform_object_detection(image, model):
    """Perform object detection using Faster R-CNN."""
    if model is None:
        return None, []
    
    try:
        # Prepare image for model
        transform = T.Compose([T.ToTensor()])
        img_tensor = transform(image).unsqueeze(0).to(DEVICE)
        
        # Run inference
        with torch.no_grad():
            predictions = model(img_tensor)
        
        # Process results
        pred = predictions[0]
        boxes = pred['boxes'].cpu().numpy()
        scores = pred['scores'].cpu().numpy()
        labels = pred['labels'].cpu().numpy()
        
        # Filter high confidence detections
        confidence_threshold = 0.5
        high_conf_indices = scores > confidence_threshold
        
        boxes = boxes[high_conf_indices]
        scores = scores[high_conf_indices]
        labels = labels[high_conf_indices]
        
        # Draw bounding boxes on image
        result_image = image.copy()
        draw = ImageDraw.Draw(result_image)
        
        detected_objects = []
        
        for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
            x1, y1, x2, y2 = box
            class_name = COCO_CLASSES[label] if label < len(COCO_CLASSES) else f"Class_{label}"
            
            # Skip background and N/A classes
            if class_name in ['__background__', 'N/A']:
                continue
            
            # Draw bounding box
            draw.rectangle([x1, y1, x2, y2], outline="yellow", width=3)
            
            # Draw label
            label_text = f"{class_name}: {score:.2f}"
            draw.text((x1, y1-20), label_text, fill="yellow")
            
            detected_objects.append({
                'class': class_name,
                'confidence': float(score),
                'bbox': [float(x1), float(y1), float(x2), float(y2)]
            })
        
        return result_image, detected_objects
        
    except Exception as e:
        st.error(f"Object detection failed: {str(e)}")
        return None, []

def perform_instance_segmentation(image, model):
    """Perform instance segmentation using Mask R-CNN."""
    if model is None:
        return None, []
    
    try:
        # Prepare image for model
        transform = T.Compose([T.ToTensor()])
        img_tensor = transform(image).unsqueeze(0).to(DEVICE)
        
        # Run inference
        with torch.no_grad():
            predictions = model(img_tensor)
        
        # Process results
        pred = predictions[0]
        boxes = pred['boxes'].cpu().numpy()
        scores = pred['scores'].cpu().numpy()
        labels = pred['labels'].cpu().numpy()
        masks = pred['masks'].cpu().numpy()
        
        # Filter high confidence detections
        confidence_threshold = 0.5
        high_conf_indices = scores > confidence_threshold
        
        boxes = boxes[high_conf_indices]
        scores = scores[high_conf_indices]
        labels = labels[high_conf_indices]
        masks = masks[high_conf_indices]
        
        # Create result image with masks
        result_image = np.array(image)
        mask_overlay = np.zeros_like(result_image)
        
        detected_instances = []
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
        
        for i, (box, score, label, mask) in enumerate(zip(boxes, scores, labels, masks)):
            class_name = COCO_CLASSES[label] if label < len(COCO_CLASSES) else f"Class_{label}"
            
            # Skip background and N/A classes
            if class_name in ['__background__', 'N/A']:
                continue
            
            # Apply mask with color
            color = colors[i % len(colors)]
            mask_binary = mask[0] > 0.5
            mask_overlay[mask_binary] = color
            
            detected_instances.append({
                'class': class_name,
                'confidence': float(score),
                'bbox': [float(x) for x in box],
                'area': float(np.sum(mask_binary))
            })
        
        # Blend original image with mask overlay
        alpha = 0.6
        result_image = cv2.addWeighted(result_image, 1-alpha, mask_overlay, alpha, 0)
        result_image_pil = Image.fromarray(result_image)
        
        return result_image_pil, detected_instances
        
    except Exception as e:
        st.error(f"Instance segmentation failed: {str(e)}")
        return None, []

def capture_frame_from_stream(stream_url):
    """Capture a frame from live stream URL."""
    try:
        cap = cv2.VideoCapture(stream_url)
        if not cap.isOpened():
            st.error("❌ Could not connect to stream URL. Please check the URL and try again.")
            return None
        
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            # Convert from BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Convert to PIL Image
            pil_image = Image.fromarray(frame_rgb)
            return pil_image
        else:
            st.error("❌ Could not capture frame from stream.")
            return None
    except Exception as e:
        st.error(f"❌ Error capturing frame: {str(e)}")
        return None

# ---- ENHANCED CSS STYLING WITH SMOOTH ANIMATIONS ----
def inject_professional_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
    
    /* Global Styles */
    .stApp {
        background: linear-gradient(135deg, #0F0F23 0%, #1A1A2E 25%, #16213E 50%, #0F3460 75%, #533483 100%);
        font-family: 'Inter', sans-serif;
        min-height: 100vh;
    }
    
    .main .block-container {
        padding: 2rem 1rem;
        max-width: 1200px;
    }
    
    /* Page Transition Animation */
    .page-transition {
        animation: pageSlideIn 0.8s cubic-bezier(0.25, 0.46, 0.45, 0.94);
    }
    
    @keyframes pageSlideIn {
        0% {
            opacity: 0;
            transform: translateY(50px);
        }
        100% {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Button Loading Animation */
    .button-loading {
        animation: buttonPulse 1.5s ease-in-out infinite;
    }
    
    @keyframes buttonPulse {
        0% {
            transform: scale(1);
            box-shadow: 0 4px 15px rgba(218, 112, 214, 0.4);
        }
        50% {
            transform: scale(1.05);
            box-shadow: 0 8px 25px rgba(255, 215, 0, 0.6);
        }
        100% {
            transform: scale(1);
            box-shadow: 0 4px 15px rgba(218, 112, 214, 0.4);
        }
    }
    
    /* Header Styles */
    .app-header {
        text-align: center;
        padding: 3rem 0 2rem 0;
        background: linear-gradient(135deg, rgba(139, 69, 19, 0.1) 0%, rgba(75, 0, 130, 0.15) 50%, rgba(30, 144, 255, 0.1) 100%);
        border-radius: 20px;
        margin-bottom: 2rem;
        backdrop-filter: blur(15px);
        border: 1px solid rgba(218, 112, 214, 0.2);
        box-shadow: 0 8px 32px rgba(139, 69, 19, 0.1);
        animation: fadeInUp 1s ease-out;
    }
    
    .app-title {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #FFD700 0%, #FF6B6B 25%, #4ECDC4 50%, #45B7D1 75%, #DA70D6 100%);
        background-clip: text;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
        text-shadow: 0 0 30px rgba(255, 215, 0, 0.3);
        animation: glow 2s ease-in-out infinite alternate;
    }
    
    .app-subtitle {
        font-size: 1.4rem;
        color: #B19CD9;
        font-weight: 500;
        margin-bottom: 1rem;
        animation: fadeIn 1.5s ease-out;
    }
    
    .app-description {
        font-size: 1.1rem;
        color: #9BB5FF;
        max-width: 600px;
        margin: 0 auto;
        line-height: 1.6;
        animation: fadeIn 2s ease-out;
    }
    
    /* Keyframe Animations */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes fadeIn {
        from {
            opacity: 0;
        }
        to {
            opacity: 1;
        }
    }
    
    @keyframes glow {
        from {
            text-shadow: 0 0 20px rgba(255, 215, 0, 0.3);
        }
        to {
            text-shadow: 0 0 30px rgba(255, 215, 0, 0.6), 0 0 40px rgba(255, 107, 107, 0.3);
        }
    }
    
    @keyframes pulse {
        0% {
            transform: scale(1);
        }
        50% {
            transform: scale(1.05);
        }
        100% {
            transform: scale(1);
        }
    }
    
    @keyframes slideInLeft {
        from {
            opacity: 0;
            transform: translateX(-50px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    @keyframes slideInRight {
        from {
            opacity: 0;
            transform: translateX(50px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    @keyframes bounceIn {
        0% {
            opacity: 0;
            transform: scale(0.3);
        }
        50% {
            opacity: 1;
            transform: scale(1.05);
        }
        70% {
            transform: scale(0.9);
        }
        100% {
            opacity: 1;
            transform: scale(1);
        }
    }
    
    /* Feature Cards */
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 1.5rem;
        margin: 2rem 0;
    }
    
    .feature-card {
        background: linear-gradient(135deg, rgba(218, 112, 214, 0.1) 0%, rgba(75, 0, 130, 0.15) 50%, rgba(30, 144, 255, 0.1) 100%);
        border-radius: 16px;
        padding: 2rem;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(218, 112, 214, 0.3);
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        position: relative;
        overflow: hidden;
        animation: slideInLeft 0.8s ease-out;
    }
    
    .feature-card:nth-child(even) {
        animation: slideInRight 0.8s ease-out;
    }
    
    .feature-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(135deg, rgba(255, 215, 0, 0.05) 0%, rgba(255, 107, 107, 0.05) 50%, rgba(218, 112, 214, 0.05) 100%);
        opacity: 0;
        transition: opacity 0.4s ease;
        pointer-events: none;
    }
    
    .feature-card:hover::before {
        opacity: 1;
    }
    
    .feature-card:hover {
        transform: translateY(-12px) scale(1.02);
        box-shadow: 0 25px 50px rgba(218, 112, 214, 0.3);
        border-color: rgba(255, 215, 0, 0.6);
    }
    
    .feature-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
        display: block;
        filter: hue-rotate(45deg) brightness(1.2);
        transition: transform 0.3s ease;
    }
    
    .feature-card:hover .feature-icon {
        transform: scale(1.1) rotate(5deg);
        animation: pulse 1s ease-in-out infinite;
    }
    
    .feature-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #FFD700;
        margin-bottom: 0.5rem;
        transition: color 0.3s ease;
    }
    
    .feature-card:hover .feature-title {
        color: #FF6B6B;
    }
    
    .feature-desc {
        color: #B19CD9;
        line-height: 1.5;
        transition: color 0.3s ease;
    }
    
    .feature-card:hover .feature-desc {
        color: #E0E0FF;
    }
    
    /* Upload Section */
    .upload-section {
        background: linear-gradient(135deg, rgba(218, 112, 214, 0.08) 0%, rgba(75, 0, 130, 0.12) 50%, rgba(30, 144, 255, 0.08) 100%);
        border-radius: 16px;
        padding: 2rem;
        margin: 2rem 0;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(218, 112, 214, 0.25);
        animation: fadeInUp 1s ease-out;
    }
    
    .upload-title {
        font-size: 1.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #FFD700 0%, #FF6B6B 50%, #DA70D6 100%);
        background-clip: text;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
        text-align: center;
        animation: glow 3s ease-in-out infinite alternate;
    }
    
    /* Results Section */
    .results-container {
        background: linear-gradient(135deg, rgba(218, 112, 214, 0.05) 0%, rgba(75, 0, 130, 0.08) 50%, rgba(30, 144, 255, 0.05) 100%);
        border-radius: 16px;
        padding: 2rem;
        margin: 2rem 0;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(218, 112, 214, 0.2);
        animation: bounceIn 1s ease-out;
    }
    
    .result-card {
        background: linear-gradient(135deg, rgba(218, 112, 214, 0.08) 0%, rgba(75, 0, 130, 0.12) 50%, rgba(30, 144, 255, 0.08) 100%);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid rgba(218, 112, 214, 0.2);
        transition: all 0.3s ease;
        animation: slideInLeft 0.6s ease-out;
    }
    
    .result-card:hover {
        transform: translateX(10px);
        box-shadow: 0 10px 25px rgba(218, 112, 214, 0.2);
    }
    
    .result-title {
        font-size: 1.25rem;
        font-weight: 600;
        background: linear-gradient(135deg, #FFD700 0%, #4ECDC4 50%, #DA70D6 100%);
        background-clip: text;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    
    .caption-text {
        font-size: 1.1rem;
        color: #B19CD9;
        background: linear-gradient(135deg, rgba(218, 112, 214, 0.1) 0%, rgba(75, 0, 130, 0.15) 100%);
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #FFD700;
        margin: 1rem 0;
        transition: all 0.3s ease;
        animation: fadeIn 1s ease-out;
    }
    
    .caption-text:hover {
        transform: translateX(5px);
        border-left-width: 6px;
        box-shadow: 0 5px 15px rgba(255, 215, 0, 0.2);
    }
    
    /* Enhanced Button Animations */
    .stButton > button {
        background: linear-gradient(135deg, #FFD700 0%, #FF6B6B 25%, #4ECDC4 50%, #45B7D1 75%, #DA70D6 100%);
        color: #0F0F23;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        box-shadow: 0 4px 15px rgba(218, 112, 214, 0.4);
        position: relative;
        overflow: hidden;
        cursor: pointer;
    }
    
    .stButton > button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.3), transparent);
        transition: left 0.5s;
    }
    
    .stButton > button:hover::before {
        left: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px) scale(1.05);
        box-shadow: 0 12px 30px rgba(255, 215, 0, 0.5);
        filter: brightness(1.2);
        animation: pulse 0.6s ease-in-out;
    }
    
    .stButton > button:active {
        transform: translateY(-1px) scale(1.02);
        box-shadow: 0 6px 20px rgba(255, 215, 0, 0.4);
    }
    
    /* Camera Section */
    .camera-section {
        background: linear-gradient(135deg, rgba(78, 205, 196, 0.08) 0%, rgba(69, 183, 209, 0.12) 50%, rgba(30, 144, 255, 0.08) 100%);
        border-radius: 16px;
        padding: 2rem;
        margin: 2rem 0;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(78, 205, 196, 0.25);
        animation: fadeInUp 1.2s ease-out;
    }
    
    .camera-preview {
        border: 3px solid rgba(255, 215, 0, 0.3);
        border-radius: 12px;
        background: linear-gradient(135deg, rgba(218, 112, 214, 0.05) 0%, rgba(75, 0, 130, 0.08) 100%);
        padding: 1rem;
        transition: all 0.3s ease;
        animation: bounceIn 1s ease-out;
    }
    
    .camera-preview:hover {
        border-color: rgba(255, 215, 0, 0.6);
        transform: scale(1.02);
        box-shadow: 0 10px 25px rgba(255, 215, 0, 0.2);
    }
    
    /* Live Video Stream Styles */
    .live-video-container {
        background: linear-gradient(135deg, rgba(218, 112, 214, 0.05) 0%, rgba(75, 0, 130, 0.08) 100%);
        border: 2px solid rgba(255, 215, 0, 0.3);
        border-radius: 12px;
        padding: 1rem;
        margin: 1rem 0;
        animation: fadeIn 1s ease-out;
        transition: all 0.3s ease;
    }
    
    .live-video-container:hover {
        border-color: rgba(255, 215, 0, 0.6);
        box-shadow: 0 8px 20px rgba(255, 215, 0, 0.2);
    }
    
    .live-caption-overlay {
        background: linear-gradient(135deg, rgba(0, 0, 0, 0.8) 0%, rgba(75, 0, 130, 0.9) 100%);
        color: #FFD700;
        padding: 1rem;
        border-radius: 8px;
        margin-top: 1rem;
        font-size: 1.1rem;
        font-weight: 600;
        text-align: center;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 215, 0, 0.3);
        animation: slideInLeft 0.8s ease-out;
        transition: all 0.3s ease;
    }
    
    .live-caption-overlay:hover {
        transform: scale(1.02);
        box-shadow: 0 5px 15px rgba(255, 215, 0, 0.3);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom file uploader */
    .stFileUploader > div {
        background: linear-gradient(135deg, rgba(218, 112, 214, 0.1) 0%, rgba(75, 0, 130, 0.15) 100%);
        border: 2px dashed rgba(255, 215, 0, 0.4);
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        transition: all 0.3s ease;
        animation: fadeIn 1s ease-out;
    }
    
    .stFileUploader > div:hover {
        border-color: rgba(255, 215, 0, 0.7);
        background: linear-gradient(135deg, rgba(218, 112, 214, 0.15) 0%, rgba(75, 0, 130, 0.2) 100%);
        transform: scale(1.02);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: linear-gradient(135deg, rgba(218, 112, 214, 0.1) 0%, rgba(75, 0, 130, 0.15) 100%);
        color: #B19CD9;
        border-radius: 10px;
        padding: 0.5rem 1rem;
        border: 1px solid rgba(218, 112, 214, 0.2);
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(218, 112, 214, 0.3);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #FFD700 0%, #FF6B6B 50%, #DA70D6 100%);
        color: #0F0F23;
        transform: translateY(-3px);
        box-shadow: 0 8px 20px rgba(255, 215, 0, 0.4);
    }
    
    /* Object detection box styles */
    .detection-stats {
        background: linear-gradient(135deg, rgba(218, 112, 214, 0.1) 0%, rgba(75, 0, 130, 0.15) 100%);
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border: 1px solid rgba(255, 215, 0, 0.3);
        transition: all 0.3s ease;
        animation: slideInRight 0.8s ease-out;
    }
    
    .detection-stats:hover {
        transform: translateX(-5px);
        box-shadow: 0 5px 15px rgba(255, 215, 0, 0.2);
    }
    
    .detection-item {
        color: #B19CD9;
        margin: 0.5rem 0;
        padding: 0.5rem;
        background: rgba(255, 215, 0, 0.1);
        border-radius: 4px;
        transition: all 0.3s ease;
    }
    
    .detection-item:hover {
        background: rgba(255, 215, 0, 0.2);
        transform: translateX(5px);
        color: #E0E0FF;
    }
    
    /* Loading animations */
    .stSpinner > div {
        border-color: #FFD700 transparent #FF6B6B transparent !important;
        animation: spin 1s linear infinite, pulse 2s ease-in-out infinite alternate !important;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    </style>
    """, unsafe_allow_html=True)

# ---- UTILITY FUNCTIONS ----
def create_download_link(img, filename, text):
    """Create a download link for processed images."""
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_data = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:image/png;base64,{img_data}" download="{filename}" style="color: #FFD700; text-decoration: none; font-weight: 500; transition: all 0.3s ease;">📥 {text}</a>'
    return href

def validate_and_open_image(image_data):
    """Enhanced image validation with better error handling."""
    try:
        image = Image.open(image_data).convert("RGB")
        
        # Image size optimization
        max_size = 1024
        if max(image.size) > max_size:
            ratio = max_size / max(image.size)
            new_size = (int(image.width * ratio), int(image.height * ratio))
            image = image.resize(new_size, Image.LANCZOS)
            st.info(f"🔄 Image resized to {new_size[0]}x{new_size[1]} for optimal processing")
        
        return image
    except UnidentifiedImageError:
        st.error("❌ The uploaded file is not a valid image or is corrupted.")
        return None
    except Exception as e:
        st.error(f"❌ Error processing image: {str(e)}")
        return None

# ---- MAIN PROCESSING FUNCTION ----
def process_image_with_ai(image, unet_model, maskrcnn_model):
    """
    Processes an image using AI models for captioning and segmentation.
    Returns a dictionary with all results.
    """
    results = {}
    progress_bar = st.progress(0, text="Initializing AI processing...")

    try:
        # Step 1: Generate Caption
        progress_bar.progress(15, text="🤖 Generating AI caption...")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            image.save(tmp.name, format="PNG")
            results['caption'] = generate_caption(tmp.name)
        
        # Step 2: Edge Detection
        progress_bar.progress(35, text="📐 Performing edge detection...")
        results['edges'] = perform_edge_detection(image)
        
        # Step 3: Object Detection
        progress_bar.progress(55, text="🎯 Detecting objects...")
        obj_img, obj_data = perform_object_detection(image, object_detection_model)
        results['object_detection'] = {
            'image': obj_img,
            'objects': obj_data
        }
        
        # Step 4: Instance Segmentation
        progress_bar.progress(75, text="👁️ Performing instance segmentation...")
        inst_img, inst_data = perform_instance_segmentation(image, maskrcnn_model)
        results['instance_segmentation'] = {
            'image': inst_img,
            'instances': inst_data
        }
        
        # Step 5: Semantic Segmentation (U-Net)
        progress_bar.progress(90, text="🧠 Performing semantic segmentation (U-Net)...")
        if unet_model is not None:
            image_for_unet = image.resize(IMAGE_SIZE, Image.LANCZOS)
            img_tensor_unet = T.ToTensor()(image_for_unet).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                output = unet_model(img_tensor_unet)
                pred_mask = torch.argmax(output, dim=1).squeeze().cpu().numpy().astype(np.uint8)
            
            # Create colorful segmentation mask
            palette = np.random.randint(0, 255, (N_CLASSES, 3), dtype=np.uint8)
            palette[0] = [0, 0, 0]  # Set background to black
            color_mask = palette[pred_mask]
            mask_img = Image.fromarray(color_mask).resize(image.size, Image.NEAREST)
            
            # Create overlay
            alpha = 0.6
            overlay_img = Image.blend(image, mask_img, alpha)
            
            results['unet_segmentation'] = {
                'mask': mask_img,
                'overlay': overlay_img
            }
        
        progress_bar.progress(100, text="✅ Analysis Complete!")
        time.sleep(1)
        progress_bar.empty()

        return results
        
    except Exception as e:
        st.error(f"❌ An error occurred during AI processing: {str(e)}")
        logging.error(f"AI Processing Error: {e}", exc_info=True)
        progress_bar.empty()
        return None

# ---- LIVE STREAMING FUNCTIONS ----
class LiveStreamProcessor:
    def __init__(self, stream_url):
        self.stream_url = stream_url
        self.cap = None
        self.running = False
        self.frame_queue = queue.Queue(maxsize=2)
        self.caption_queue = queue.Queue(maxsize=1)
        self.current_caption = "Starting live analysis..."
        
    def start_stream(self):
        """Start the video stream."""
        try:
            self.cap = cv2.VideoCapture(self.stream_url)
            if not self.cap.isOpened():
                return False
            
            # Set buffer size to reduce latency
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.running = True
            return True
        except Exception as e:
            st.error(f"❌ Error starting stream: {str(e)}")
            return False
    
    def stop_stream(self):
        """Stop the video stream."""
        self.running = False
        if self.cap:
            self.cap.release()
    
    def get_frame(self):
        """Get the latest frame from the stream."""
        if not self.cap or not self.running:
            return None
        
        ret, frame = self.cap.read()
        if ret:
            # Convert from BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return Image.fromarray(frame_rgb)
        return None
    
    def process_caption_async(self, image):
        """Process caption in a separate thread."""
        def caption_worker():
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                    image.save(tmp.name, format="PNG")
                    caption = generate_caption(tmp.name)
                    
                # Update caption queue (non-blocking)
                try:
                    self.caption_queue.put_nowait(caption)
                except queue.Full:
                    pass  # Skip if queue is full
                    
            except Exception as e:
                pass  # Silent fail for background processing
        
        # Start caption processing in background
        caption_thread = Thread(target=caption_worker, daemon=True)
        caption_thread.start()
    
    def get_latest_caption(self):
        """Get the latest caption if available."""
        try:
            while not self.caption_queue.empty():
                self.current_caption = self.caption_queue.get_nowait()
        except queue.Empty:
            pass
        return self.current_caption

# ---- APP STATE MANAGEMENT ----
if "page" not in st.session_state:
    st.session_state.page = "welcome"
if "processed_results" not in st.session_state:
    st.session_state.processed_results = None
if "camera_capture" not in st.session_state:
    st.session_state.camera_capture = None
if "camera_connected" not in st.session_state:
    st.session_state.camera_connected = False
if "live_captioning_active" not in st.session_state:
    st.session_state.live_captioning_active = False
if "live_stream_processor" not in st.session_state:
    st.session_state.live_stream_processor = None
if "transitioning" not in st.session_state:
    st.session_state.transitioning = False

# ---- UI PAGES ----

def show_welcome_page():
    """Renders the landing/welcome page."""
    inject_professional_css()
    
    st.markdown("""
    <div class="app-header">
        <div class="app-title">Vision AI Suite</div>
        <div class="app-subtitle">Professional Computer Vision Platform</div>
        <div class="app-description">
            Transform your images with cutting-edge artificial intelligence. 
            Generate intelligent captions, perform precise segmentation, and analyze visual content 
            with professional-grade accuracy powered by state-of-the-art neural networks.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    _, col2, _ = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 Launch Vision AI", key="enter_app", use_container_width=True):
            # Add transition effect
            st.session_state.transitioning = True
            
            # Show loading animation
            with st.spinner("🚀 Launching Vision AI..."):
                time.sleep(1.5)  # Smooth transition delay
            
            st.session_state.page = "main"
            st.session_state.transitioning = False
            st.rerun()
    
    st.markdown("""
    <div class="feature-grid">
        <div class="feature-card">
            <div class="feature-icon">🧠</div>
            <div class="feature-title">AI-Powered Captions</div>
            <div class="feature-desc">
                Generate intelligent, contextual descriptions for any image using advanced 
                transformer models trained on millions of image-text pairs.
            </div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">🎯</div>
            <div class="feature-title">Object Detection</div>
            <div class="feature-desc">
                Identify and locate objects in images using state-of-the-art 
                Faster R-CNN architecture with COCO dataset training.
            </div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">👁</div>
            <div class="feature-title">Instance Segmentation</div>
            <div class="feature-desc">
                Advanced object detection and instance segmentation using Mask R-CNN 
                for identifying and localizing multiple objects in images.
            </div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">🧠</div>
            <div class="feature-title">U-Net Segmentation</div>
            <div class="feature-desc">
                Precise pixel-level classification using custom U-Net architecture 
                trained on COCO dataset with multiple object categories.
            </div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">📐</div>
            <div class="feature-title">Edge Detection</div>
            <div class="feature-desc">
                Multiple edge detection algorithms including Canny, Sobel, and Laplacian 
                for detailed structural analysis and feature extraction.
            </div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">📊</div>
            <div class="feature-title">Comprehensive Analysis</div>
            <div class="feature-desc">
                Complete computer vision pipeline with downloadable results, 
                detailed statistics, and professional-grade visualization tools.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def show_main_page():
    """Renders the main application interface."""
    inject_professional_css()
    
    # Add page transition class
    st.markdown('<div class="page-transition">', unsafe_allow_html=True)
    
    _, col2, _ = st.columns([1, 6, 1])
    with col2:
        st.markdown("""
        <div style="text-align: center;">
            <h1 style="background: linear-gradient(135deg, #FFD700 0%, #FF6B6B 50%, #DA70D6 100%); background-clip: text; -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 700; margin: 0;">🤖 Vision AI Processing</h1>
            <p style="color: #B19CD9; margin: 0.5rem 0;">Upload an image or capture from camera for AI analysis</p>
        </div>
        """, unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["📸 Image Upload", "📹 Stream Capture", "🎬 Live Captioning"])
    
    with tab1:
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Choose an image file", 
            type=["jpg", "jpeg", "png", "bmp", "webp"],
            label_visibility="collapsed"
        )
        
        if uploaded_file:
            orig_image = validate_and_open_image(uploaded_file)
            if orig_image:
                _, col2, _ = st.columns([1, 2, 1])
                with col2:
                    st.image(orig_image, caption='📷 Original Image', use_column_width=True)
                
                if st.button("🚀 Analyze with AI", key="process_upload", use_container_width=True):
                    results = process_image_with_ai(orig_image, unet, maskrcnn)
                    if results:
                        st.session_state.processed_results = results
                        st.success("✅ Analysis complete! Check results below.")
                    else:
                        st.error("❌ AI processing failed. Please check the logs or try another image.")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        show_camera_capture_interface()
    
    with tab3:
        show_live_captioning_interface()
    
    if st.session_state.processed_results:
        show_analysis_results(st.session_state.processed_results)
    
    st.markdown('</div>', unsafe_allow_html=True)  # Close page-transition div

def show_camera_capture_interface():
    """Renders the camera capture UI with live stream URL."""
    st.markdown('<div class="camera-section">', unsafe_allow_html=True)
    st.markdown('<div class="upload-title">📹 Live Stream Frame Capture</div>', unsafe_allow_html=True)
    
    # Stream URL input
    stream_url = st.text_input(
        "📡 Enter Live Stream URL", 
        placeholder="http://your-stream-url.com/stream.mjpg",
        help="Enter the URL of your mobile camera stream (e.g., IP Webcam app stream URL)"
    )
    
    # Check for a previously captured image to avoid re-running the camera
    if st.session_state.get('camera_capture'):
        img = st.session_state.camera_capture
        st.markdown('<div class="camera-preview">', unsafe_allow_html=True)
        st.image(img, caption='📸 Captured Frame from Stream', use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
             if st.button("🤖 Analyze Captured Frame", key="process_camera", use_container_width=True):
                results = process_image_with_ai(img, unet, maskrcnn)
                if results:
                    st.session_state.processed_results = results
                    st.success("✅ Frame analysis complete! Check results below.")
                else:
                    st.error("❌ AI processing failed for the captured frame.")
        with col2:
            if st.button("🔄 Capture New Frame", use_container_width=True):
                st.session_state.camera_capture = None
                st.rerun()

    else:
        # Capture frame from stream URL
        if stream_url:
            if st.button("📸 Capture Frame from Stream", use_container_width=True):
                with st.spinner("Capturing frame from stream..."):
                    captured_image = capture_frame_from_stream(stream_url)
                    if captured_image:
                        st.session_state.camera_capture = captured_image
                        st.success("✅ Frame captured successfully!")
                        st.rerun()
        else:
            st.info("📱 Please enter your mobile camera stream URL above to capture frames.")
                
    st.markdown('</div>', unsafe_allow_html=True)

def show_live_captioning_interface():
    """Renders the live captioning interface with continuous video stream and automatic captions."""
    st.markdown('<div class="camera-section">', unsafe_allow_html=True)
    st.markdown('<div class="upload-title">🎬 Live Camera with Real-time AI Captions</div>', unsafe_allow_html=True)
    
    # Camera input method selection
    st.markdown("### 📹 Camera Input Method")
    camera_method = st.radio(
        "Choose your camera input:",
        ["📱 Mobile Camera Stream (URL)", "💻 Built-in Camera"],
        key="camera_method",
        horizontal=True
    )
    
    if camera_method == "📱 Mobile Camera Stream (URL)":
        # Stream URL input for live captioning
        live_stream_url = st.text_input(
            "📡 Enter Live Stream URL for Real-time Analysis", 
            placeholder="http://your-stream-url.com/stream.mjpg",
            help="Enter the URL of your mobile camera stream for live AI analysis",
            key="live_stream_url"
        )
        
        # Live analysis controls
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🎬 Start Live Camera with Captions", key="start_live_stream", use_container_width=True):
                if live_stream_url:
                    st.session_state.live_captioning_active = True
                    st.session_state.live_stream_processor = LiveStreamProcessor(live_stream_url)
                    if st.session_state.live_stream_processor.start_stream():
                        st.success("🚀 Live camera with captions started!")
                        st.rerun()
                    else:
                        st.error("❌ Could not connect to camera stream. Please check your URL.")
                        st.session_state.live_captioning_active = False
                else:
                    st.error("❌ Please enter a stream URL first.")
        
        with col2:
            if st.button("⏹️ Stop Live Camera", key="stop_live_stream", use_container_width=True):
                if st.session_state.live_stream_processor:
                    st.session_state.live_stream_processor.stop_stream()
                st.session_state.live_captioning_active = False
                st.session_state.live_stream_processor = None
                st.info("⏹️ Live camera stopped.")
                st.rerun()
        
        # Display live video stream with captions
        if st.session_state.live_captioning_active and st.session_state.live_stream_processor:
            st.markdown("---")
            st.markdown("## 🔴 LIVE CAMERA WITH AI CAPTIONS")
            
            # Create containers for live content
            video_container = st.empty()
            caption_container = st.empty()
            
            # Continuous video stream loop
            frame_count = 0
            while st.session_state.live_captioning_active:
                # Get latest frame
                current_frame = st.session_state.live_stream_processor.get_frame()
                
                if current_frame is not None:
                    # Display live video feed
                    video_container.image(
                        current_frame, 
                        caption=f"🎥 LIVE CAMERA FEED - {time.strftime('%H:%M:%S')}", 
                        use_column_width=True
                    )
                    
                    # Process captions every 3rd frame to balance performance with responsiveness
                    if frame_count % 3 == 0:
                        st.session_state.live_stream_processor.process_caption_async(current_frame)
                    
                    # Get and display latest caption
                    latest_caption = st.session_state.live_stream_processor.get_latest_caption()
                    caption_container.markdown(
                        f'<div class="live-caption-overlay">🤖 AI Caption: {latest_caption}</div>', 
                        unsafe_allow_html=True
                    )
                    
                    frame_count += 1
                    
                    # Small delay to prevent overwhelming the system
                    time.sleep(0.03)  # ~30 FPS
                    
                else:
                    st.error("❌ Lost connection to camera stream.")
                    st.session_state.live_captioning_active = False
                    break
        
        elif not live_stream_url:
            st.info("📱 Please enter your mobile camera stream URL above to start live captioning.")
    
    else:  # Built-in Camera
        st.markdown("### 💻 Built-in Camera Live Captioning")
        
        # Use Streamlit's camera input for built-in camera
        camera_photo = st.camera_input("📸 Take a photo for AI analysis")
        
        if camera_photo is not None:
            # Process the captured photo
            image = Image.open(camera_photo)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.image(image, caption="📸 Captured Photo", use_column_width=True)
            
            with col2:
                if st.button("🤖 Generate Caption", key="caption_photo", use_container_width=True):
                    with st.spinner("🤖 Generating AI caption..."):
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                            image.save(tmp.name, format="PNG")
                            caption = generate_caption(tmp.name)
                            
                        st.markdown(
                            f'<div class="caption-text">🤖 AI Caption: {caption}</div>', 
                            unsafe_allow_html=True
                        )
                
                if st.button("🔍 Full AI Analysis", key="analyze_photo", use_container_width=True):
                    results = process_image_with_ai(image, unet, maskrcnn)
                    if results:
                        st.session_state.processed_results = results
                        st.success("✅ Analysis complete! Check results below.")
    
    st.markdown('</div>', unsafe_allow_html=True)

def show_analysis_results(results):
    """Displays all the analysis results in a formatted way."""
    st.markdown('<div class="results-container">', unsafe_allow_html=True)
    st.markdown("## 📊 Comprehensive Analysis Results")
    
    # Caption Results
    if results.get('caption'):
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.markdown('<div class="result-title">🤖 AI Generated Caption</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="caption-text">{results["caption"]}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Edge Detection Results
    if results.get('edges'):
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.markdown('<div class="result-title">📐 Edge Detection Analysis</div>', unsafe_allow_html=True)
        
        edge_cols = st.columns(3)
        edges = results['edges']
        
        with edge_cols[0]:
            st.image(edges['canny'], caption="Canny Edge Detection", use_column_width=True)
            st.markdown(create_download_link(edges['canny'], "canny_edges.png", "Download Canny"), unsafe_allow_html=True)
        
        with edge_cols[1]:
            st.image(edges['sobel'], caption="Sobel Edge Detection", use_column_width=True)
            st.markdown(create_download_link(edges['sobel'], "sobel_edges.png", "Download Sobel"), unsafe_allow_html=True)
        
        with edge_cols[2]:
            st.image(edges['laplacian'], caption="Laplacian Edge Detection", use_column_width=True)
            st.markdown(create_download_link(edges['laplacian'], "laplacian_edges.png", "Download Laplacian"), unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Object Detection Results
    if results.get('object_detection'):
        obj_result = results['object_detection']
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.markdown('<div class="result-title">🎯 Object Detection</div>', unsafe_allow_html=True)
        
        if obj_result['image']:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.image(obj_result['image'], caption="Detected Objects with Bounding Boxes", use_column_width=True)
                st.markdown(create_download_link(obj_result['image'], "object_detection.png", "Download Result"), unsafe_allow_html=True)
            
            with col2:
                if obj_result['objects']:
                    st.markdown('<div class="detection-stats">', unsafe_allow_html=True)
                    st.markdown(f"**Objects Detected: {len(obj_result['objects'])}**")
                    for obj in obj_result['objects']:
                        st.markdown(f'<div class="detection-item">• {obj["class"]}: {obj["confidence"]:.2f}</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.info("No objects detected with high confidence.")
        else:
            st.info("Object detection was not performed (model unavailable).")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Instance Segmentation Results
    if results.get('instance_segmentation'):
        inst_result = results['instance_segmentation']
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.markdown('<div class="result-title">👁️ Instance Segmentation</div>', unsafe_allow_html=True)
        
        if inst_result['image']:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.image(inst_result['image'], caption="Instance Segmentation with Colored Masks", use_column_width=True)
                st.markdown(create_download_link(inst_result['image'], "instance_segmentation.png", "Download Result"), unsafe_allow_html=True)
            
            with col2:
                if inst_result['instances']:
                    st.markdown('<div class="detection-stats">', unsafe_allow_html=True)
                    st.markdown(f"**Instances Segmented: {len(inst_result['instances'])}**")
                    for inst in inst_result['instances']:
                        st.markdown(f'<div class="detection-item">• {inst["class"]}: {inst["confidence"]:.2f}<br>Area: {inst["area"]:.0f} pixels</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.info("No instances detected with high confidence.")
        else:
            st.info("Instance segmentation was not performed (model unavailable).")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # U-Net Semantic Segmentation Results
    if results.get('unet_segmentation'):
        unet_result = results['unet_segmentation']
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.markdown('<div class="result-title">🧠 U-Net Semantic Segmentation</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if unet_result['mask']:
                st.image(unet_result['mask'], caption="U-Net Segmentation Mask", use_column_width=True)
                st.markdown(create_download_link(unet_result['mask'], "unet_mask.png", "Download Mask"), unsafe_allow_html=True)
            else:
                st.info("U-Net segmentation mask not available.")
        
        with col2:
            if unet_result['overlay']:
                st.image(unet_result['overlay'], caption="U-Net Segmentation Overlay", use_column_width=True)
                st.markdown(create_download_link(unet_result['overlay'], "unet_overlay.png", "Download Overlay"), unsafe_allow_html=True)
            else:
                st.info("U-Net segmentation overlay not available.")
        
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ---- MAIN APP ROUTING ----
def main():
    """Main function to route between pages."""
    if st.session_state.page == "welcome":
        show_welcome_page()
    else:
        show_main_page()

if __name__ == "__main__":
    main()