import os
import uuid
import urllib.request
from PIL import Image
import numpy as np
import io
from datetime import datetime
import time

import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.preprocessing.image import load_img, img_to_array # Still useful for initial loading if needed

from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

# Define f1_metric as a dummy for model loading
def f1_metric(y_true, y_pred):
    return 0.0

# Define your class labels
classes = ['Acne', 'Eczema', 'Moles', 'Psoriasis', 'Rosacea',
           'Seborrheic Keratoses', 'Sun Damage', 'Vitiligo', 'Warts']

app = Flask(__name__, template_folder='.', static_folder='static')

# Configure CORS
CORS(app, resources={
    r"/*": {
        "origins": ["http://192.168.100.8:3000", "http://localhost:3000", "http://127.0.0.1:3000"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

# Configure upload folder for saved images
UPLOAD_FOLDER = 'uploads'
STATIC_IMAGES_FOLDER = 'static/images'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_IMAGES_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['STATIC_IMAGES_FOLDER'] = STATIC_IMAGES_FOLDER

# Define the target input shape for your model (HEIGHT, WIDTH, CHANNELS)
TARGET_IMAGE_HEIGHT = 75
TARGET_IMAGE_WIDTH = 100
TARGET_IMAGE_SHAPE = (TARGET_IMAGE_HEIGHT, TARGET_IMAGE_WIDTH, 3)

# --- MODEL LOADING AND COMBINATION ---
print("Loading TensorFlow model (feature extractor)...")
try:
    # Load the base model (your feature extractor like InceptionV3)
    base_model = load_model(os.path.join(os.path.dirname(os.path.abspath(__file__)) , 'incepti.h5'),
                            custom_objects={'f1_metric': f1_metric},
                            compile=False)
    print("Base model loaded successfully. Inspecting output shape before modification...")
    print(f"Original Base model input shape: {base_model.input_shape}")
    print(f"Original Base model output shape: {base_model.output_shape}")

    # Create an Input layer that matches the DESIRED input shape for your combined model
    # (None, 75, 100, 3)
    inputs = Input(shape=TARGET_IMAGE_SHAPE)

    # Note: If your 'incepti.h5' is specifically a pre-trained model like InceptionV3
    # it likely expects (224, 224, 3) or (299, 299, 3). Resizing to (75, 100) before
    # feeding it to such a model might lead to poor performance unless 'incepti.h5'
    # was *specifically trained* on (75, 100) images.
    # If incepti.h5 is InceptionV3, you might need to insert an appropriate resizing layer
    # or ensure your base_model itself is flexible or adapted.
    # For now, we'll connect it directly as requested, but keep this performance note in mind.

    # Connect the inputs to the base_model
    # This might require careful handling if base_model expects a different input shape
    # than TARGET_IMAGE_SHAPE. If base_model expects (224,224,3), and you feed (75,100,3),
    # it WILL cause an error or unexpected behavior.
    # The safest way is to ensure base_model's input layer can adapt or resize.
    # For a general base_model, a Lambda layer for resizing might be needed if shapes mismatch:
    # from tensorflow.keras.layers import Lambda
    # x = Lambda(lambda image: tf.image.resize(image, (base_model.input_shape[1], base_model.input_shape[2])))(inputs)
    # x = base_model(x, training=False)

    # Assuming 'incepti.h5' can accept or be adapted to the new input shape 'TARGET_IMAGE_SHAPE'
    x = base_model(inputs, training=False)

    # Ensure output is flattened if it's not already 2D (batch, features)
    if len(x.shape) > 2:
        x = Flatten()(x)

    # Add your custom classification layers
    x = Dense(units=128, activation='relu')(x)
    predictions = Dense(units=len(classes), activation='softmax')(x)

    # Create the final combined model with the desired input shape
    model = Model(inputs=inputs, outputs=predictions)

    print(f"Combined model created with input shape: {model.input_shape}")
    print(f"Combined model output shape: {model.output_shape}")
    model.summary()
except Exception as e:
    print(f"Error loading or building model: {e}")
    model = None # Set model to None if loading fails to prevent further errors


# Allowed file extensions for uploads
ALLOWED_EXT = set(['jpg', 'jpeg', 'png', 'jfif'])
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXT

def preprocess_image_for_prediction(image_bytes):
    """Preprocess image for model prediction (matches the combined model's input)"""
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    # Resize to the TARGET_IMAGE_SHAPE (WIDTH, HEIGHT) for PIL.Image.resize
    img = img.resize((TARGET_IMAGE_WIDTH, TARGET_IMAGE_HEIGHT))
    img_array = np.array(img)
    img_array = img_array / 255.0
    return np.expand_dims(img_array, axis=0)

def predict_condition(img_array):
    """Run prediction on the preprocessed image array"""
    if model is None:
        raise Exception("Model not loaded. Cannot make prediction.")

    predictions = model.predict(img_array)
    predicted_class_idx = np.argmax(predictions[0])
    confidence = float(predictions[0][predicted_class_idx]) * 100

    return classes[predicted_class_idx], round(confidence, 2)

# MongoDB setup
mongo_uri = os.getenv('MONGODB_URI', 'mongodb+srv://neder:r81n5pEFft90n2RV@cluster0.ccquyi0.mongodb.net/endoskin?retryWrites=true&w=majority&appName=Cluster0')
try:
    from pymongo import MongoClient # Import MongoClient here to ensure it's loaded only if needed
    client = MongoClient(mongo_uri, connectTimeoutMS=60000, socketTimeoutMS=120000)
    # Ping the MongoDB server to check if connection is successful
    client.admin.command('ping')
    print("MongoDB connection established successfully.")
except Exception as e:
    print(f"Failed to connect to MongoDB: {e}")
    client = None # Set client to None if connection fails

# Helper function to save analysis result to MongoDB with retries
def save_to_mongodb(data, max_retries=3):
    if client is None:
        print("MongoDB client not available. Skipping save.")
        return None

    db = client.get_database('endoskin')
    analysis_collection = db.analysis_results
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
            time.sleep(2 ** attempts)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/predict_image', methods=['POST'])
def predict_image_endpoint():
    # Handle direct file upload
    if 'file' in request.files:
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No selected file'}), 400

        if not allowed_file(file.filename):
            return jsonify({'success': False, 'error': 'Invalid file type. Only JPG, JPEG, PNG, JFIF allowed.'}), 400

        try:
            filename = secure_filename(f"{datetime.now().timestamp()}_{file.filename}")
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            print(f"File saved successfully at: {filepath}")

            with open(filepath, 'rb') as f:
                image_bytes = f.read()

            processed_img_array = preprocess_image_for_prediction(image_bytes)

            condition, confidence = predict_condition(processed_img_array)

            result_data = {
                'originalFilename': file.filename,
                'savedPath': filepath,
                'predictedCondition': condition,
                'confidence': confidence,
                'type': 'upload',
                'status': 'success',
                'createdAt': datetime.now()
            }

            try:
                save_to_mongodb(result_data)
            except Exception as e:
                print(f"Error saving to MongoDB: {e}")
                result_data['status'] = 'failed_to_save'

            response_data = {
                'success': True,
                'prediction': {
                    'condition': condition,
                    'confidence': f"{confidence}%",
                    'timestamp': datetime.now().isoformat()
                },
                'imageUrl': f"/uploads/{filename}"
            }
            return jsonify(response_data), 200

        except Exception as e:
            print(f"Error during file upload prediction: {e}")
            return jsonify({'success': False, 'error': str(e)}), 500

    # Handle image via URL link
    elif 'link' in request.form:
        link = request.form.get('link')
        if not link:
            return jsonify({'success': False, 'error': 'No image link provided'}), 400

        try:
            resource = urllib.request.urlopen(link, timeout=10)
            image_bytes = resource.read()

            unique_filename = str(uuid.uuid4())
            content_type = resource.info().get_content_type()
            ext = 'jpg'
            if 'image/' in content_type:
                ext = content_type.split('/')[-1]
            filename = f"{unique_filename}.{ext}"

            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            with open(filepath, "wb") as output_file:
                output_file.write(image_bytes)
            print(f"Image from URL saved successfully at: {filepath}")

            processed_img_array = preprocess_image_for_prediction(image_bytes)

            condition, confidence = predict_condition(processed_img_array)

            result_data = {
                'originalLink': link,
                'savedPath': filepath,
                'predictedCondition': condition,
                'confidence': confidence,
                'type': 'link',
                'status': 'success',
                'createdAt': datetime.now()
            }
            try:
                save_to_mongodb(result_data)
            except Exception as e:
                print(f"Error saving to MongoDB: {e}")
                result_data['status'] = 'failed_to_save'

            response_data = {
                'success': True,
                'prediction': {
                    'condition': condition,
                    'confidence': f"{confidence}%",
                    'timestamp': datetime.now().isoformat()
                },
                'imageUrl': f"/uploads/{filename}"
            }
            return jsonify(response_data), 200

        except urllib.error.URLError as e:
            print(f"URL error: {e}")
            return jsonify({'success': False, 'error': f"Could not retrieve image from URL: {e.reason}"}), 400
        except Exception as e:
            print(f"Error during URL prediction: {e}")
            return jsonify({'success': False, 'error': f"Failed to process image from link: {str(e)}"}), 500
    else:
        return jsonify({'success': False, 'error': 'No image file or link provided in the request'}), 400

if __name__ == '__main__':
    # Ensure directories exist before starting the app
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['STATIC_IMAGES_FOLDER'], exist_ok=True)
    app.run(host='0.0.0.0', port=5000, debug=True)