if not os.path.exists("yolov5"):
    os.system("git clone https://github.com/ultralytics/yolov5")
import urllib.request

MODEL_URL = "https://huggingface.co/ArjunDcw/helmet-model/resolve/main/best.pt"

if not os.path.exists("best.pt"):
    print("Downloading model...")
    urllib.request.urlretrieve(MODEL_URL, "best.pt")
import streamlit as st
import numpy as np
import cv2
from PIL import Image

# YOLOv8
from ultralytics import YOLO

# YOLOv5
import sys
import os


sys.path.append("yolov5")

from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes
from utils.torch_utils import select_device
import torch

st.title("MTV Traffic Violation Detection")

# Load models
vehicle_model = YOLO("yolov8n.pt")

device = select_device('cpu')
helmet_model = DetectMultiBackend("best.pt", device=device)
helmet_model.eval()

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    st.image(image, caption="Uploaded Image", use_container_width=True)

    # -------------------------------
    # Step 1: Detect objects (YOLOv8)
    # -------------------------------
    results = vehicle_model(image_np)
    names_vehicle = vehicle_model.names

    motorcycles = []
    persons = []

    for box in results[0].boxes:
        cls = int(box.cls[0])
        label = names_vehicle[cls]
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        if label == "motorcycle":
            motorcycles.append((x1, y1, x2, y2))
        elif label == "person":
            persons.append((x1, y1, x2, y2))

    # -------------------------------
    # Step 2: Helmet detection (FULL IMAGE)
    # -------------------------------
    img_full = cv2.resize(image_np, (640, 640))
    img_full = img_full.transpose((2, 0, 1))
    img_full = np.ascontiguousarray(img_full)

    img_full = torch.from_numpy(img_full).to(device)
    img_full = img_full.float() / 255.0
    img_full = img_full.unsqueeze(0)

    pred_full = helmet_model(img_full)
    pred_full = non_max_suppression(pred_full, 0.25, 0.45)

    names_helmet = helmet_model.names
    helmet_detections = []

    for det in pred_full:
        if len(det):
            det[:, :4] = scale_boxes(img_full.shape[2:], det[:, :4], image_np.shape).round()
            for *xyxy, conf, cls in det:
                label = names_helmet[int(cls)]
                x1h, y1h, x2h, y2h = map(int, xyxy)
                helmet_detections.append((label, x1h, y1h, x2h, y2h))

    # -------------------------------
    # Step 3: Match person → motorcycle → helmet
    # -------------------------------
    output_img = image_np.copy()
    violation_found = False

    for (x1, y1, x2, y2) in motorcycles:

        # Draw motorcycle
        cv2.rectangle(output_img, (x1, y1), (x2, y2), (0,255,0), 2)

        # Find rider (person overlapping motorcycle)
        rider_box = None
        for (x1p, y1p, x2p, y2p) in persons:
            if (x1p < x2 and x2p > x1 and y1p < y2 and y2p > y1):
                rider_box = (x1p, y1p, x2p, y2p)
                break

        status = "No Helmet ❌"
        color = (0,0,255)

        if rider_box:
            rx1, ry1, rx2, ry2 = rider_box

            # Draw rider box (optional for demo clarity)
            cv2.rectangle(output_img, (rx1, ry1), (rx2, ry2), (255,0,0), 2)

            for hlabel, x1h, y1h, x2h, y2h in helmet_detections:
                if (x1h < rx2 and x2h > rx1 and y1h < ry2 and y2h > ry1):
                    if "helmet" in hlabel:
                        status = "Helmet ✅"
                        color = (0,255,0)
                        break
                    elif "no-helmet" in hlabel:
                        status = "No Helmet ❌"
                        color = (0,0,255)
                        violation_found = True

        if "No Helmet" in status:
            violation_found = True

        # Label
        cv2.putText(output_img, status, (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # -------------------------------
    # Output
    # -------------------------------
    st.image(output_img, caption="Detection Result", use_container_width=True)

    if violation_found:
        st.error("Violation Detected ❌")
    else:
        st.success("No Violation ✅")
