import io
import os
import cv2
import torch
import base64
import logging
import traceback
import numpy as np
import torchvision
from PIL import Image
from torchvision import transforms
from flask import Flask, render_template, request, jsonify
from torchvision.models.detection import  fasterrcnn_mobilenet_v3_large_fpn

# Flask app initialization
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model variables
resnet_model = None
faster_rcnn_model = None


# Model Loading Functions
def load_resnet_model() -> torch.nn.Module:
    """Loads and prepares the ResNet18 model for uniform detection."""
    num_classes = 2  # TRUE/FALSE classes
    model = torchvision.models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, num_classes)
    model_path = os.getenv("RESNET_MODEL_PATH", "clothing_model.pth")
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    return model


def load_fasterrcnn_model() -> torch.nn.Module:
    """Loads and prepares the Faster R-CNN model for tie detection."""
    num_classes = 2  # Adjust for the dataset
    model = fasterrcnn_mobilenet_v3_large_fpn(num_classes=num_classes)
    model_path = os.getenv("FASTERRCNN_MODEL_PATH", "./fasterrcnn_model3.pth")
    logger.info(f"Loading model from {model_path}")
    try:
        model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading the model: {e}")
    model.eval()
    return model


def detect_dominant_color(image: Image.Image, box: list) -> np.ndarray:
    """Detects the dominant color in a specified region."""
    xmin, ymin, xmax, ymax = [int(coord) for coord in box]
    logger.info(f"Cropping region: {(xmin, ymin, xmax, ymax)}")

    image_pil = Image.fromarray(image)  # Convert NumPy array to PIL Image
    cropped_region = image_pil.crop((xmin, ymin, xmax, ymax))  # Crop the ROI
    image_np = np.array(cropped_region)  # Convert cropped region back to NumPy array
    mean_color = image_np.mean(axis=(0, 1))  # Average across width and height
    logger.info(f"Dominant color: {mean_color}")
    return mean_color.astype(int)


def is_grey(color: np.ndarray, tolerance: int = 20) -> bool:
    """Determines if a given color is grey."""
    r, g, b = color
    return abs(r - g) < tolerance and abs(g - b) < tolerance and abs(r - b) < tolerance


def is_white_shirt(image: np.ndarray, hsv_threshold: float = 0.4, lab_threshold: float = 0.4) -> bool:
    """Determines if a cropped region contains a white shirt or similar tones like light blue."""
    if image is None or image.size == 0:
        return False
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    lower_white_hsv = np.array([0, 0, 150])
    upper_white_hsv = np.array([180, 50, 255])

    lower_white_lab = np.array([190, 120, 110])
    upper_white_lab = np.array([255, 150, 140])

    white_mask_hsv = cv2.inRange(hsv_image, lower_white_hsv, upper_white_hsv)
    white_mask_lab = cv2.inRange(lab_image, lower_white_lab, upper_white_lab)

    white_ratio_hsv = cv2.countNonZero(white_mask_hsv) / (image.size / 3)
    white_ratio_lab = cv2.countNonZero(white_mask_lab) / (image.size / 3)

    return white_ratio_hsv > hsv_threshold or white_ratio_lab > lab_threshold


def is_blue_background(image: np.ndarray, threshold: float = 0.1) -> bool:
    """Determines if the background color is predominantly blue."""
    resized_image = cv2.resize(image, (300, 300))
    hsv_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2HSV)

    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([130, 255, 255])

    blue_mask = cv2.inRange(hsv_image, lower_blue, upper_blue)
    blue_ratio = cv2.countNonZero(blue_mask) / (resized_image.shape[0] * resized_image.shape[1])
    return blue_ratio > threshold


def predict(image: Image.Image, model: torch.nn.Module, transform: transforms.Compose) -> int:
    """Predicts whether the image satisfies uniform compliance."""
    image_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
    return predicted.item()


# Preprocessing function
def preprocess_image(file: bytes) -> tuple:
    """Preprocesses the uploaded image."""
    image = Image.open(io.BytesIO(file)).convert("RGB")
    image_np = np.array(image)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    encoded_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return image, image_bgr, encoded_image


def edges(image: Image.Image) -> str:
    """Detects edges in the image and checks for edges in the margins."""
    image_np = np.array(image)
    gray_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray_image, (5, 5), 0), 100, 200)

    height, width = edges.shape
    margin = 10

    regions = {
        "Top": (0, 0, width, margin),
        "Bottom": (0, height - margin, width, height),
        "Left": (0, 0, margin, height),
        "Right": (width - margin, 0, width, height)
    }

    threshold = 5.0

    for name, (x1, y1, x2, y2) in regions.items():
        region = edges[y1:y2, x1:x2]
        total_pixels = region.size
        edge_pixels = np.sum(region > 0)
        edge_percentage = (edge_pixels / total_pixels) * 100

        if edge_percentage > threshold:
            logger.info(f"FAIL: {name} edge detected with {edge_pixels} pixels ({edge_percentage:.2f}%).")
            return "มีขอบขาว"
    else:
        logger.info("PASS: No edges detected in any region.")
        return "ไม่มีขอบขาว"


def analyze_image(image: Image.Image, image_bgr: np.ndarray, tie_transform: transforms.Compose, analysis_type: str) -> dict:
    """Analyzes the image for uniform compliance, optionally skipping certain checks."""
    results = {"Edge": "", "shirt": "", "face": "", "background": "", "shirt_color": "", "tie": "", "status": ""}

    # Always check edges
    results["Edge"] = edges(image)

    # Predict shirt compliance
    predicted_class = predict(image, resnet_model, uniform_transform)
    results["shirt"] = "เครื่องแบบถูกต้อง" if predicted_class else "เครื่องแบบไม่ถูกต้อง"

    # Check for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    results["face"] = "ตรวจพบใบหน้า" if len(faces) > 0 else "ตรวจไม่พบใบหน้า"

    # Skip background detection if analysis_type is 2
    if analysis_type == "1":
        results["background"] = "พื้นหลังเป็นสีฟ้า" if is_blue_background(image_bgr) else "พื้นหลังไม่ใช่สีฟ้า"
    else:
        results["background"] = "ไม่ได้ตรวจสอบพื้นหลัง"

    # Detect shirt color only if faces are detected
    if len(faces) > 0:
        x, y, w, h = faces[0]
        shirt_region = image_bgr[y + h: y + 2 * h, x: x + w]
        results["shirt_color"] = "เสื้อเป็นสีขาว" if is_white_shirt(shirt_region) else "เสื้อไม่ใช่สีขาว"

    # Tie detection logic
    image_tensor = tie_transform(image).unsqueeze(0)
    with torch.no_grad():
        prediction = faster_rcnn_model(image_tensor)

    boxes = prediction[0].get('boxes', torch.empty(0)).cpu()
    scores = prediction[0].get('scores', torch.empty(0)).cpu()

    detected_ties = []
    image_np = np.array(image)
    for box, score in zip(boxes, scores):
        if score > 0.8:
            color = detect_dominant_color(image_np, box)
            if is_grey(color):
                detected_ties.append("เน็กไทร์เป็นสีเทา")
            else:
                detected_ties.append("เน็กไทร์ไม่ใช่สีเทา")
    results["tie"] = detected_ties[0] if detected_ties else "ไม่พบเน็กไทร์"

    # Determine overall status
    if any([
        results["Edge"] == "มีขอบขาว",
        results["shirt"] == "เครื่องแบบไม่ถูกต้อง",
        results["face"] == "ตรวจไม่พบใบหน้า",
        results["background"] == "พื้นหลังไม่ใช่สีฟ้า" and analysis_type == "1",
        results["shirt_color"] == "เสื้อไม่ใช่สีขาว",
        results["tie"] == "เน็กไทร์ไม่ใช่สีเทา"
    ]):
        results["status"] = "รูปภาพไม่ผ่านเกณฑ์"
    else:
        results["status"] = "รูปภาพผ่านเกณฑ์"

    return results


# Define the routes
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict/", methods=["POST"])
def predict_image():
    try:
        file = request.files["file"]
        analysis_type = request.form.get("analysis_type", "1")  # Default to 1

        image, image_bgr, encoded_image = preprocess_image(file.read())
        
        # Pass analysis_type to the analyze_image function
        results = analyze_image(image, image_bgr, tie_transform, analysis_type)
        results["uploaded_image"] = f"data:image/jpeg;base64,{encoded_image}"
        return render_template("result.html", results=results)
    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)})


# Startup function
def startup():
    global resnet_model, faster_rcnn_model
    try:
        logger.info("Starting model loading process...")
        resnet_model = load_resnet_model()
        faster_rcnn_model = load_fasterrcnn_model()
        logger.info("Models loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        raise RuntimeError("Failed to load models.")

# Initialize models on startup
startup()

# Define the transforms
uniform_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

tie_transform = transforms.Compose([
    transforms.ToTensor()
])

if __name__ == "__main__":
    app.run(debug=True)
