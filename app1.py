from flask import Flask, request, jsonify, make_response
from flask_cors import CORS, cross_origin
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
from pymongo import MongoClient
from datetime import datetime
import time
from werkzeug.utils import secure_filename

app = Flask(__name__)
# Configure CORS more permissively for debugging
CORS(app, resources={
    r"/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["*"]
    }
})

@app.before_request
def log_request_info():
    print(f"\nIncoming request: {request.method} {request.path}")
    print("Headers:", request.headers)
    if request.method == 'POST':
        print("Form data:", request.form)
        print("Files:", request.files)

# Class labels mapping
class_labels = {
    0: 'Acne',
    1: 'Eczema',
    2: 'Moles',
    3: 'Psoriasis',
    4: 'Rosacea',
    5: 'Seborrheic Keratoses',
    6: 'Sun Damage',
    7: 'Vitiligo',
    8: 'Warts'
}

# Load model at startup
print("Loading TensorFlow model...")
model = tf.keras.models.load_model('incepti.h5')
print("Model loaded successfully")

# MongoDB setup
mongo_uri = os.getenv('MONGODB_URI', 'mongodb+srv://neder:r81n5pEFft90n2RV@cluster0.ccquyi0.mongodb.net/endoskin?retryWrites=true&w=majority&appName=Cluster0')
client = MongoClient(mongo_uri, connectTimeoutMS=60000, socketTimeoutMS=120000)
db = client.get_database('endoskin')
analysis_collection = db.analysis_results

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def preprocess_image(image_bytes):
    """Preprocess image for model prediction"""
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    # Resize to match model's expected input shape (width, height)
    img = img.resize((100, 75))  
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

def save_to_mongodb(data, max_retries=3):
    """Save analysis result to MongoDB with retries"""
    attempts = 0
    while attempts < max_retries:
        try:
            result = analysis_collection.insert_one(data)
            return str(result.inserted_id)
        except Exception as e:
            attempts += 1
            print(f"Save attempt {attempts} failed: {e}")
            if attempts >= max_retries:
                raise
            time.sleep(2 ** attempts)  # Exponential backoff

@app.route('/')
def health_check():
    """Endpoint for server connectivity checks"""
    return jsonify({
        'status': 'ok',
        'message': 'Server is running',
        'timestamp': datetime.now().isoformat()
    }), 200, {'Content-Type': 'application/json'}

@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    """Handle image upload and prediction"""
    print("Received predict request")  # Debug log
    
    if 'image' not in request.files:
        print("No image in request.files")  # Debug log
        return jsonify({
            'success': False, 
            'error': 'No image provided'
        }), 400, {'Content-Type': 'application/json'}

    file = request.files['image']
    print(f"Received file: {file.filename}")  # Debug log
    if file.filename == '':
        return jsonify({
            'success': False, 
            'error': 'No selected file'
        }), 400, {'Content-Type': 'application/json'}

    try:
        # Save file temporarily
        filename = secure_filename(f"{datetime.now().timestamp()}_{file.filename}")
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Read and preprocess image
        with open(filepath, 'rb') as f:
            image_bytes = f.read()
        processed_img = preprocess_image(image_bytes)

        # Make prediction
        predictions = model.predict(processed_img)
        predicted_class = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class])
# --- ADD THESE PRINT STATEMENTS ---
        print(f"Raw predictions output shape: {predictions.shape}")
        print(f"Raw predictions output (first row): {predictions[0]}")
        # Prepare result for MongoDB
        result_data = {
            'imagePath': filepath,
            'predictedCondition': class_labels[predicted_class],
            'confidence': confidence,
            'status': 'success',
            'createdAt': datetime.now()
        }

        # Save to MongoDB
        try:
            save_to_mongodb(result_data)
        except Exception as e:
            print(f"Failed to save to MongoDB: {e}")
            result_data['status'] = 'failed_to_save'

        # Prepare response
        response = {
            'success': True,
            'prediction': {
                'condition': class_labels[predicted_class],
                'confidence': f"{confidence * 100:.2f}%",
                'timestamp': datetime.now().isoformat()
            },
            'imageUrl': f"/uploads/{filename}"
        }

        response = make_response(jsonify(response), 200)
        response.headers['Content-Type'] = 'application/json'
        response.headers['Access-Control-Allow-Origin'] = '*'
        return response

    except Exception as e:
        print(f"Prediction error: {e}")
        response = make_response(jsonify({
            'success': False,
            'error': str(e)
        }), 500)
        response.headers['Content-Type'] = 'application/json'
        response.headers['Access-Control-Allow-Origin'] = '*'
        return response

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
