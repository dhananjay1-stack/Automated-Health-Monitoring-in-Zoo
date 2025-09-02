import os
import torch
import torch.nn.functional as F
from flask import Flask, request, render_template, redirect, url_for, jsonify
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from torchvision import transforms

# Import your model
from train2 import TigerBehaviorModel

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "static/uploads"
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 100MB max file size

# Ensure upload folder exists
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -------------------------------
# Load both models
# -------------------------------
try:
    binary_model = TigerBehaviorModel(num_classes=2)
    binary_model.load_state_dict(torch.load("best_tiger_model_2_classes.pth", map_location=device, weights_only=True))
    binary_model = binary_model.to(device)
    binary_model.eval()
    print("✅ Binary model loaded successfully")
except Exception as e:
    print(f"❌ Error loading binary model: {e}")
    binary_model = None

try:
    multi_model = TigerBehaviorModel(num_classes=15)
    multi_model.load_state_dict(torch.load("best_tiger_model_15_classes.pth", map_location=device, weights_only=True))
    multi_model = multi_model.to(device)
    multi_model.eval()
    print("✅ Multi-class model loaded successfully")
except Exception as e:
    print(f"❌ Error loading multi-class model: {e}")
    multi_model = None

# -------------------------------
# Class maps
# -------------------------------
binary_class_map = {0: "Tiger_Normal", 1: "Tiger_Abnormal"}

multi_class_id_to_name = {
    0: 'Dehydration or Heat Stroke',
    1: 'Digestive Issues',
    2: 'Eye Injury',
    3: 'Injured_Tiger',
    4: 'Lethargy, Apathy, Unresponsive, and Listless Tiger',
    5: 'Neurological Issues',
    6: 'Nutritional_Deficiencies',
    7: 'Oral or Dental Issues or Respiratory distress',
    8: 'Skin Desease or irritation_Tiger',
    9: 'Sress_Frustation',
    10: 'Tremors or Seizures',
    11: 'underweightness or emaciation',
    12: 'Weakness',
    13: 'Zoochosis_stereotypic behavior',
    14: 'Zoonotic Disease Behavior'
}

# -------------------------------
# Helper: Extract frames from video - FIXED
# -------------------------------
def extract_frames(video_path, clip_len=8, frame_size=(224, 224)):
    """
    Extract frames from video and return in the correct format for the model
    Expected output shape: [C, T, H, W] where C=channels, T=time, H=height, W=width
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return None
    
    frames = []
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Video info: {total_frames} frames, {fps} fps")
    
    # Calculate frame indices to sample uniformly across the video
    if total_frames <= clip_len:
        frame_indices = list(range(total_frames))
    else:
        frame_indices = np.linspace(0, total_frames - 1, clip_len, dtype=int)
    
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            print(f"Warning: Could not read frame {frame_idx}")
            continue
            
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize frame
        frame_resized = cv2.resize(frame_rgb, frame_size)
        
        # Convert to tensor and normalize to [0, 1]
        frame_tensor = torch.FloatTensor(frame_resized / 255.0)
        
        # Convert from HWC to CHW
        frame_tensor = frame_tensor.permute(2, 0, 1)  # [C, H, W]
        
        frames.append(frame_tensor)
        frame_count += 1
    
    cap.release()
    
    if len(frames) == 0:
        print("Error: No frames extracted from video")
        return None
    
    # Pad with last frame if necessary
    while len(frames) < clip_len:
        frames.append(frames[-1].clone())
        print(f"Padding frame {len(frames)-1}")
    
    # Stack frames along time dimension: [T, C, H, W]
    frames_tensor = torch.stack(frames)  # [T, C, H, W]
    
    # Rearrange to [C, T, H, W] as expected by the model
    frames_tensor = frames_tensor.permute(1, 0, 2, 3)  # [C, T, H, W]
    
    print(f"Final tensor shape: {frames_tensor.shape}")  # Should be [3, clip_len, 224, 224]
    return frames_tensor

# -------------------------------
# Inference function
# -------------------------------
def predict_tiger_behavior(frames_tensor, binary_model, multi_model):
    """
    Perform hierarchical prediction on video frames
    """
    if frames_tensor is None:
        return {"error": "No frames to process"}
    
    # Add batch dimension: [C, T, H, W] -> [1, C, T, H, W]
    frames_batch = frames_tensor.unsqueeze(0).to(device)
    
    try:
        # Step 1: Binary classification
        with torch.no_grad():
            binary_output, video_feats, pose_feats = binary_model(frames_batch)
            binary_probs = F.softmax(binary_output, dim=1)
            binary_pred = torch.argmax(binary_probs, dim=1).item()
            binary_confidence = torch.max(binary_probs, dim=1)[0].item()
            binary_label = binary_class_map[binary_pred]
        
        print(f"Binary prediction: {binary_label} (confidence: {binary_confidence:.3f})")
        
        # Step 2: Multi-class if abnormal
        if binary_label == "Tiger_Abnormal" and multi_model is not None:
            with torch.no_grad():
                multi_output, _, _ = multi_model(frames_batch)
                multi_probs = F.softmax(multi_output, dim=1)
                multi_pred = torch.argmax(multi_probs, dim=1).item()
                multi_confidence = torch.max(multi_probs, dim=1)[0].item()
                subclass_label = multi_class_id_to_name.get(multi_pred, f"Unknown_{multi_pred}")
            
            final_prediction = f"Tiger_Abnormal - {subclass_label}"
            final_confidence = min(binary_confidence, multi_confidence)
            
            print(f"Multi-class prediction: {subclass_label} (confidence: {multi_confidence:.3f})")
            
        else:
            final_prediction = "Tiger_Normal"
            final_confidence = binary_confidence
        
        return {
            "prediction": final_prediction,
            "confidence": final_confidence,
            "binary_prediction": binary_label,
            "binary_confidence": binary_confidence
        }
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        return {"error": f"Prediction failed: {str(e)}"}

# -------------------------------
# Routes
# -------------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Check if models are loaded
        if binary_model is None:
            return render_template("index.html", 
                                 prediction="Error: Binary model not loaded", 
                                 error=True)
        
        # Check if file was uploaded
        if "file" not in request.files:
            return render_template("index.html", 
                                 prediction="No file uploaded", 
                                 error=True)
        
        file = request.files["file"]
        if file.filename == "":
            return render_template("index.html", 
                                 prediction="No file selected", 
                                 error=True)
        
        # Check file extension
        allowed_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}
        if not any(file.filename.lower().endswith(ext) for ext in allowed_extensions):
            return render_template("index.html", 
                                 prediction="Please upload a valid video file (mp4, avi, mov, etc.)", 
                                 error=True)
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        
        try:
            file.save(file_path)
            print(f"File saved to: {file_path}")
        except Exception as e:
            return render_template("index.html", 
                                 prediction=f"Error saving file: {str(e)}", 
                                 error=True)
        
        # Extract frames
        print("Extracting frames...")
        frames = extract_frames(file_path, clip_len=8)
        
        if frames is None:
            # Clean up file
            if os.path.exists(file_path):
                os.remove(file_path)
            return render_template("index.html", 
                                 prediction="Could not extract frames from video", 
                                 error=True)
        
        # Make prediction
        print("Making prediction...")
        result = predict_tiger_behavior(frames, binary_model, multi_model)
        
        # Clean up uploaded file (optional - comment out if you want to keep files)
        # if os.path.exists(file_path):
        #     os.remove(file_path)
        
        if "error" in result:
            return render_template("index.html", 
                                 prediction=result["error"], 
                                 error=True)
        else:
            return render_template("index.html", 
                                 prediction=result["prediction"],
                                 confidence=f"{result['confidence']:.1%}",
                                 binary_prediction=result["binary_prediction"],
                                 binary_confidence=f"{result['binary_confidence']:.1%}",
                                 video_path=file_path)
    
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    """
    Alternative predict endpoint (redirects to main index)
    """
    return index()

@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    API endpoint for programmatic access
    """
    if binary_model is None:
        return jsonify({"error": "Models not loaded"}), 500
    
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400
    
    # Save file
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    
    try:
        file.save(file_path)
        
        # Process video
        frames = extract_frames(file_path, clip_len=8)
        if frames is None:
            return jsonify({"error": "Could not extract frames"}), 400
        
        # Make prediction
        result = predict_tiger_behavior(frames, binary_model, multi_model)
        
        # Clean up
        if os.path.exists(file_path):
            os.remove(file_path)
        
        return jsonify(result)
        
    except Exception as e:
        # Clean up on error
        if os.path.exists(file_path):
            os.remove(file_path)
        return jsonify({"error": str(e)}), 500

@app.route("/health")
def health_check():
    """
    Health check endpoint
    """
    status = {
        "status": "healthy",
        "binary_model_loaded": binary_model is not None,
        "multi_model_loaded": multi_model is not None,
        "device": str(device)
    }
    return jsonify(status)

@app.errorhandler(413)
def too_large(e):
    return render_template("index.html", 
                         prediction="File too large. Please upload a video smaller than 100MB.", 
                         error=True)

@app.errorhandler(500)
def internal_error(e):
    return render_template("index.html", 
                         prediction="Internal server error occurred.", 
                         error=True)

if __name__ == "__main__":
    print("🐅 Tiger Behavior Classification Flask App")
    print(f"Device: {device}")
    print(f"Binary Model: {'✅ Loaded' if binary_model else '❌ Not Loaded'}")
    print(f"Multi Model: {'✅ Loaded' if multi_model else '❌ Not Loaded'}")
    print("Starting Flask server...")
    
    app.run(debug=True, host='0.0.0.0', port=5000)