import os
import cv2
import numpy as np
import pytesseract
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from PIL import Image
import io
from ultralytics import YOLO

# --- Configuration ---

# Path to Tesseract executable (if not in system PATH)
# On Windows, it might be:
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Define your class names (must match your data.yaml)
CLASS_NAMES = ['helmet', 'no_helmet', 'number_plate']
# Get class IDs from names
try:
    NO_HELMET_ID = CLASS_NAMES.index('no_helmet')
    NUMBER_PLATE_ID = CLASS_NAMES.index('number_plate')
except ValueError as e:
    print(f"Error: {e}. Make sure 'no_helmet' and 'number_plate' are in CLASS_NAMES.")
    exit()

# Intersection-over-Union (IoU) threshold for associating a plate with a person
IOU_THRESHOLD = 0.02

# --- Flask App Initialization ---
app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing

# --- Load YOLOv8 Model ---
try:
    # EDIT THIS PATH to point to your trained 'best.pt' file
    MODEL_PATH = 'runs/detect/train8/weights/best.pt' 
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Please update the MODEL_PATH variable in app.py.")
    
    print(f"Loading model from {MODEL_PATH}...")
    model = YOLO(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please ensure the model path is correct and 'ultralytics' is installed.")
    exit()

# --- Helper Functions ---

def preprocess_for_ocr(image_crop: np.ndarray) -> np.ndarray:
    """Applies preprocessing steps to a cropped image for better OCR results."""
    print("Preprocessing for OCR...")
    # Convert to grayscale
    gray = cv2.cvtColor(image_crop, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding to binarize the image
    # THRESH_OTSU automatically finds the best threshold value
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    # Optional: Apply morphological operations to remove noise
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    # opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    # Invert image (Tesseract often works better with black text on white background)
    inverted_image = 255 - thresh
    return inverted_image

def extract_plate_text(image_crop: np.ndarray) -> str:
    """Performs OCR on the preprocessed image crop."""
    try:
        # Preprocess the image
        processed_crop = preprocess_for_ocr(image_crop)
        
        # Use Tesseract to extract text
        # --psm 6: Assume a single uniform block of text.
        # -c tessedit_char_whitelist: Restrict to alphanumeric characters
        custom_config = r'--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        
        text = pytesseract.image_to_string(processed_crop, config=custom_config)
        
        # Clean up the extracted text
        cleaned_text = "".join(filter(str.isalnum, text)).upper()
        
        print(f"OCR Result: '{text}' -> Cleaned: '{cleaned_text}'")
        return cleaned_text
    except Exception as e:
        print(f"Error during OCR: {e}")
        return ""

def calculate_iou(boxA, boxB):
    """Calculates Intersection over Union (IoU) between two bounding boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

# --- Flask Routes ---

@app.route('/')
def index():
    """Serves the main HTML page."""
    # This is not strictly needed if you just open the file,
    # but it's good practice.
    return "Please open index.html in your browser."

@app.route('/predict', methods=['POST'])
def predict():
    """Handles the image upload and prediction."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        try:
            # Read image in-memory
            img_bytes = file.read()
            img = Image.open(io.BytesIO(img_bytes))
            
            # Convert PIL Image to OpenCV format (NumPy array)
            img_np = np.array(img)
            img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

            print("Running model prediction...")
            # Perform detection
            results = model.predict(img_cv)
            
            # Process results
            result = results[0]
            boxes = result.boxes.xyxy.cpu().numpy()  # Bounding boxes
            classes = result.boxes.cls.cpu().numpy()  # Class IDs

            violator_plates = []
            violators = []
            plates = []

            # Separate violators and plates
            for i in range(len(boxes)):
                cls_id = int(classes[i])
                box = boxes[i]
                
                if cls_id == NO_HELMET_ID:
                    violators.append(box)
                elif cls_id == NUMBER_PLATE_ID:
                    plates.append(box)

            print(f"Detections: {len(violators)} violators, {len(plates)} plates")

            # Associate plates with violators
            if not plates or not violators:
                print("No violations or no plates found.")
                return jsonify({'plates': []})

            for plate_box in plates:
                for violator_box in violators:
                    # Check for proximity. This is a simple check.
                    # A better check might be IoU or distance.
                    # Let's use IoU on the full image.
                    iou = calculate_iou(plate_box, violator_box)
                    
                    # This logic is basic: "is a plate near a violator?"
                    # You might want to improve this logic, e.g.,
                    # "is the plate *part of the same motorcycle* as the violator?"
                    # For now, any proximity is considered a match.
                    if iou > IOU_THRESHOLD:
                        print(f"Violation detected: Plate near violator (IoU: {iou:.4f})")
                        
                        # Crop the number plate
                        x1, y1, x2, y2 = [int(coord) for coord in plate_box]
                        plate_crop = img_cv[y1:y2, x1:x2]
                        
                        # Extract text
                        plate_text = extract_plate_text(plate_crop)
                        
                        if plate_text and plate_text not in violator_plates:
                            violator_plates.append(plate_text)
                        
                        # Once a plate is associated, break to not double-count
                        break 

            print(f"Returning detected plates: {violator_plates}")
            return jsonify({'plates': violator_plates})

        except Exception as e:
            print(f"Error during prediction: {e}")
            return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Start the Flask app
    # host='0.0.0.0' makes it accessible on your local network
    app.run(host='0.0.0.0', port=8000, debug=True)
