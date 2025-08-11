from flask import Flask, render_template, request, jsonify
# import pickle   # Uncomment when using real model
import cv2
import numpy as np
from werkzeug.utils import secure_filename
import os

# ====== CONFIG ======
UPLOAD_FOLDER = "static/uploads"
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "mp4"}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ====== LOAD MODEL ======
# model = pickle.load(open("model.pkl", "rb"))  # Uncomment when model is ready

# ====== UTILS ======
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_image(filepath):
    """ Read and preprocess image before prediction """
    img = cv2.imread(filepath)
    img_resized = cv2.resize(img, (224, 224))  # Change size as per your model
    img_array = np.expand_dims(img_resized, axis=0) / 255.0
    return img_array

# ====== ROUTES ======
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # ====== DUMMY PREDICTION FOR NOW ======
        img_array = process_image(filepath)
        # prediction = model.predict(img_array)  # Uncomment when model ready
        # label = "Normal" if prediction[0] == 0 else "Abnormal"
        label = "Dummy: Normal"  # Temporary label

        return jsonify({"label": label, "filepath": filepath})
    else:
        return jsonify({"error": "Invalid file format"}), 400

if __name__ == "__main__":
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)
