import cv2
import os
import gdown
import torch
import datetime
from torchvision import transforms
from PIL import Image
from collections import deque

from ultralytics import YOLO
from models.sequence_model import CNNLSTM
from speed_estimation import SpeedEstimator
from distance_estimation import DistanceEstimator


# =====================
# INPUT VIDEO
# =====================
INPUT_DIR = "input_videos"

if not os.path.exists(INPUT_DIR):
    raise FileNotFoundError("input_videos folder not found")

videos = [
    f for f in os.listdir(INPUT_DIR)
    if f.lower().endswith((".mp4", ".webm", ".avi", ".mov", ".mkv"))
]

if not videos:
    raise RuntimeError("No video found in input_videos folder")

print("\nAvailable videos:")
for v in videos:
    print(f"- {v}")

video_name = input("\nEnter the video filename to use (exact name): ").strip()
if video_name not in videos:
    raise ValueError("Video not found in input_videos folder")

VIDEO_PATH = os.path.join(INPUT_DIR, video_name)


# =====================
# CONFIG & DEVICE
# =====================
RISK_INTERVAL = 1
YOLO_CONF = 0.5
WINDOW_NAME = "YOLO + Overtake Risk"

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)


# =====================
# LOAD MODELS
# =====================

# YOLO (auto-downloads if missing)
yolo = YOLO("yolov8n.pt")

# CNN-LSTM model
risk_model = CNNLSTM().to(device)

model_path = "models/cnn_lstm_risk_40epoch.pth"
file_id = "1NeIMkbwoZ-wWIpK7PXM8UPpEX_a_ryG0"

os.makedirs("models", exist_ok=True)

if not os.path.exists(model_path):
    print("Model not found. Downloading...")
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, model_path, quiet=False)
    print("Download complete.")

risk_model.load_state_dict(torch.load(model_path, map_location=device))
risk_model.eval()


# =====================
# TRANSFORM
# =====================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


# =====================
# VIDEO INPUT
# =====================
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError("Error opening video file")

input_fps = cap.get(cv2.CAP_PROP_FPS) or 30
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


# =====================
# OUTPUT VIDEO
# =====================
OUTPUT_DIR = "output_videos"
os.makedirs(OUTPUT_DIR, exist_ok=True)

timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
OUTPUT_VIDEO = os.path.join(OUTPUT_DIR, f"output_{timestamp}.mp4")

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, input_fps, (width, height))


# =====================
# STATE
# =====================
frame_buffer = deque(maxlen=5)
current_risk = None
frame_count = 0

speed_estimator = SpeedEstimator()
distance_estimator = DistanceEstimator()

distance_label = "UNKNOWN"
oncoming_speed = 0.0


def classify_speed(speed):
    if speed < 2.0:
        return "SLOW"
    elif speed < 5.0:
        return "MEDIUM"
    else:
        return "FAST"


# =====================
# MAIN LOOP
# =====================
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    yolo_results = yolo(frame, conf=YOLO_CONF, verbose=False)
    annotated = frame.copy()

    oncoming_vehicle = False
    pedestrian_block = False
    pedestrian_warning = False

    frame_center_x = width // 2

    for box in yolo_results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls[0])
        label = yolo_results[0].names[cls]

        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            annotated, label, (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
        )

        box_center_x = (x1 + x2) // 2
        box_area = (x2 - x1) * (y2 - y1)

        # Distance estimation
        box_width = x2 - x1
        distance_label = distance_estimator.estimate(box_width)

        if label in ["car", "truck", "bus", "motorcycle"]:
            if box_center_x > frame_center_x:
                oncoming_vehicle = True
                center = (box_center_x, (y1 + y2) // 2)
                oncoming_speed = speed_estimator.update(center, frame_count)
                speed_label = classify_speed(oncoming_speed)

        if label == "person" and box_area > 8000:
            if box_center_x > frame_center_x:
                pedestrian_block = True
            else:
                pedestrian_warning = True


    # ---------- RISK MODEL ----------
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    tensor_img = transform(pil_img)
    frame_buffer.append(tensor_img)

    if len(frame_buffer) == 5 and frame_count % RISK_INTERVAL == 0:
        frames = torch.stack(list(frame_buffer)).unsqueeze(0).to(device)
        with torch.no_grad():
            current_risk = torch.sigmoid(risk_model(frames)).item()


    # ---------- UI ----------
    if current_risk is not None:

        if pedestrian_block:
            text, color = "UNSAFE OVERTAKE - PEDESTRIAN RIGHT", (0, 0, 255)

        elif oncoming_vehicle:
            text, color = "UNSAFE OVERTAKE - ONCOMING VEHICLE", (0, 0, 255)

        elif current_risk > 0.6:
            text, color = f"UNSAFE OVERTAKE ({current_risk:.2f})", (0, 0, 255)

        else:
            text, color = f"SAFE OVERTAKE ({current_risk:.2f})", (0, 255, 0)

        cv2.putText(
            annotated, text, (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2
        )

        if oncoming_vehicle:
            cv2.putText(
                annotated,
                f"Oncoming speed: {speed_label} ({oncoming_speed:.1f})",
                (30, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2
            )


    if oncoming_vehicle:
        cv2.putText(
            annotated,
            f"Distance: {distance_label}",
            (30, 160),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 200, 0),
            2
        )


    if pedestrian_warning:
        cv2.putText(
            annotated,
            "CAUTION: PEDESTRIAN LEFT SIDE",
            (30, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 255),
            2
        )


    # ---------- OUTPUT ----------
    out.write(annotated)

    display_frame = cv2.resize(annotated, (1280, 720)) if width > 1280 else annotated
    cv2.imshow(WINDOW_NAME, display_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
out.release()
cv2.destroyAllWindows()