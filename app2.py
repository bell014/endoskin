from flask import Flask, request, jsonify, make_response, render_template
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
from sklearn.metrics import f1_score

app = Flask(__name__)

# Configure CORS
CORS(app, resources={
    r"/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["*"]
    }
})

# Model loading with custom metric
def f1_metric(y_true, y_pred):
    return f1_score(y_true, y_pred)

# Load model
print("Loading TensorFlow model...")
model = tf.keras.models.load_model('incepti.h5', custom_objects={'f1_metric': f1_metric})
print("Model loaded successfully")

# Class labels mapping
classes = ['Acne', 'Eczema', 'Moles', 'Psoriasis', 'Rosacea',
          'Seborrheic Keratoses', 'Sun Damage', 'Vitiligo', 'Warts']

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

def predict_condition(img_array):
    """Run prediction using the new approach"""
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    confidence = float(predictions[0][predicted_class]) * 100
    return classes[predicted_class], round(confidence, 2)

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
def home():
    return render_template("index.html")

@app.route('/predict_image', methods=['POST'])
@cross_origin()
def predict():
    """Handle image upload and prediction"""
    if 'image' not in request.files:
        return jsonify({
            'success': False, 
            'error': 'No image provided'
        }), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({
            'success': False, 
            'error': 'No selected file'
        }), 400

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
        condition, confidence = predict_condition(processed_img)

        # Prepare result for MongoDB
        result_data = {
            'imagePath': filepath,
            'predictedCondition': condition,
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
                'condition': condition,
                'confidence': f"{confidence}%",
                'timestamp': datetime.now().isoformat()
            },
            'imageUrl': f"/uploads/{filename}"
        }

        return make_response(jsonify(response), 200)

    except Exception as e:
        print(f"Prediction error: {e}")
        return make_response(jsonify({
            'success': False,
            'error': str(e)
        }), 500)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
