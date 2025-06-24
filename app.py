import os
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"
import streamlit as st
from PIL import Image
import tempfile
import cv2
import numpy as np
from inference import generate_caption  # Your captioning function

st.title("Image Captioning Demo")

# --- File uploader ---
uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png", "bmp", "tiff", "tif", "gif", "webp"]
)

# --- IP Webcam streaming ---
st.write("Or stream from an IP Webcam (Android IP Webcam app, etc.):")
stream_url = st.text_input("http://192.168.31.9:8080/video:")

# Use session state to store the last frame and caption
if 'ipwebcam_frame' not in st.session_state:
    st.session_state['ipwebcam_frame'] = None
if 'ipwebcam_caption' not in st.session_state:
    st.session_state['ipwebcam_caption'] = None

if stream_url:
    cap = cv2.VideoCapture(stream_url)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        st.warning("Failed to fetch frame. Check your stream URL and network.")
        st.session_state['ipwebcam_frame'] = None
    else:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st.session_state['ipwebcam_frame'] = frame_rgb
        st.image(frame_rgb, caption="Live IP Webcam Frame", channels="RGB", use_column_width=True)

        # Button to capture and caption
        if st.button("Capture Frame and Generate Caption (IP Webcam)"):
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmpfile:
                img = Image.fromarray(frame_rgb)
                img.save(tmpfile.name)
                caption = generate_caption(tmpfile.name)
                st.session_state['ipwebcam_caption'] = caption

        # Display caption if available
        if st.session_state['ipwebcam_caption']:
            st.write("*Caption:*", st.session_state['ipwebcam_caption'])

# --- Display and caption for uploaded file ---
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    if img.mode != "RGB":
        img = img.convert("RGB")
    st.image(img, caption='Uploaded Image', use_container_width=True)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        img.save(tmp.name, format="PNG")
        if st.button("Generate Caption for Uploaded Image"):
            caption = generate_caption(tmp.name)
            st.write("*Caption:*", caption)

