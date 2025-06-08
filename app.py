import os
import uuid
import urllib.request
import gc
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

# Configure TensorFlow for optimal performance
tf.config.optimizer.set_jit(True)  # Enable XLA compilation
tf.config.threading.set_inter_op_parallelism_threads(2)
tf.config.threading.set_intra_op_parallelism_threads(2)

def convert_to_tflite(model_path):
    """Convert Keras model to TensorFlow Lite"""
    tflite_model_path = os.path.join(os.path.dirname(model_path), 'model.tflite')
    
    if not os.path.exists(tflite_model_path):
        print("Converting model to TensorFlow Lite format...")
        model = load_model(model_path, custom_objects={'f1_metric': f1_metric}, compile=False)
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        with open(tflite_model_path, 'wb') as f:
            f.write(tflite_model)
    
    return tflite_model_path

# --- MODEL LOADING AND COMBINATION ---
print("Loading and optimizing TensorFlow model...")
try:
    # Convert to TensorFlow Lite
    tflite_model_path = convert_to_tflite(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'incepti.h5')
    )
    
    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
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

from multiprocessing import Pool, cpu_count

def preprocess_single_image(image_bytes):
    """Helper function for parallel processing"""
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((TARGET_IMAGE_WIDTH, TARGET_IMAGE_HEIGHT))
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

def preprocess_image_for_prediction(image_bytes):
    """Preprocess image for model prediction (matches the combined model's input)"""
    return preprocess_single_image(image_bytes)

def preprocess_image_batch_parallel(image_batch):
    """Process batch of images in parallel"""
    with Pool(processes=min(cpu_count(), 4)) as pool:
        results = pool.map(preprocess_single_image, image_batch)
    return results

def predict_condition(img_array):
    """Run optimized prediction using TensorFlow Lite"""
    if not hasattr(predict_condition, 'interpreter'):
        raise Exception("Model not loaded. Cannot make prediction.")

    # Get model references
    interpreter = predict_condition.interpreter
    input_details = predict_condition.input_details
    output_details = predict_condition.output_details

    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], img_array)
    
    # Run inference
    interpreter.invoke()
    
    # Get output
    predictions = interpreter.get_tensor(output_details[0]['index'])
    predicted_class_idx = np.argmax(predictions[0])
    confidence = float(predictions[0][predicted_class_idx]) * 100

    return classes[predicted_class_idx], round(confidence, 2)

# Store model references for prediction function
predict_condition.interpreter = interpreter
predict_condition.input_details = input_details
predict_condition.output_details = output_details

# Performance monitoring
def log_performance(start_time, operation):
    duration = time.time() - start_time
    print(f"{operation} completed in {duration:.2f} seconds")

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
def save_to_mongodb(data, max_retries=3, bulk=False):
    if client is None:
        print("MongoDB client not available. Skipping save.")
        return None

    db = client.get_database('endoskin')
    analysis_collection = db.analysis_results
    
    if bulk and isinstance(data, list):
        # Handle bulk operations
        attempts = 0
        while attempts < max_retries:
            try:
                result = analysis_collection.insert_many(data)
                return [str(id) for id in result.inserted_ids]
            except Exception as e:
                attempts += 1
                print(f"Bulk save attempt {attempts} failed: {e}")
                if attempts >= max_retries:
                    raise
                time.sleep(2 ** attempts)
    else:
        # Handle single document operations
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

# Configuration
DEFAULT_BATCH_SIZE = 5  # Number of images to process at once
MAX_BATCH_SIZE = 10     # Maximum allowed batch size

def process_image_batch(image_batch):
    """Process a batch of images and return predictions"""
    results = []
    try:
        # Separate images and filenames for parallel processing
        images = [item[0] for item in image_batch]
        filenames = [item[1] for item in image_batch]
        
        # Process images in parallel
        processed_arrays = preprocess_image_batch_parallel(images)
        
        # Make predictions
        for i, img_array in enumerate(processed_arrays):
            try:
                condition, confidence = predict_condition(img_array)
                results.append({
                    'filename': filenames[i],
                    'condition': condition,
                    'confidence': confidence,
                    'image_array': img_array  # Keep reference for cleanup
                })
            except Exception as e:
                results.append({
                    'filename': filenames[i],
                    'error': str(e)
                })
    except Exception as e:
        # Fallback to sequential processing if parallel fails
        print(f"Parallel processing failed, falling back to sequential: {str(e)}")
        for image_bytes, filename in image_batch:
            try:
                processed_img_array = preprocess_image_for_prediction(image_bytes)
                condition, confidence = predict_condition(processed_img_array)
                results.append({
                    'filename': filename,
                    'condition': condition,
                    'confidence': confidence,
                    'image_array': processed_img_array
                })
            except Exception as e:
                results.append({
                    'filename': filename,
                    'error': str(e)
                })
    return results

def cleanup_batch_resources(batch_results):
    """Explicitly clean up resources from batch processing"""
    for result in batch_results:
        if 'image_array' in result:
            del result['image_array']  # Remove reference to numpy array
    gc.collect()  # Force garbage collection

def process_url_batch(url_batch):
    """Process a batch of URLs and return predictions"""
    results = []
    for link in url_batch:
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
            
            processed_img_array = preprocess_image_for_prediction(image_bytes)
            condition, confidence = predict_condition(processed_img_array)
            
            results.append({
                'link': link,
                'filename': filename,
                'condition': condition,
                'confidence': confidence,
                'image_array': processed_img_array
            })
        except Exception as e:
            results.append({
                'link': link,
                'error': str(e)
            })
    return results

@app.route('/predict_image', methods=['POST'])
def predict_image_endpoint():
    batch_size = min(int(request.form.get('batch_size', DEFAULT_BATCH_SIZE)), MAX_BATCH_SIZE)
    
    # Handle direct file uploads (single or multiple)
    if 'file' in request.files:
        files = request.files.getlist('file')
        if not files or all(f.filename == '' for f in files):
            return jsonify({'success': False, 'error': 'No selected files'}), 400

        # Validate all files first
        for file in files:
            if not allowed_file(file.filename):
                return jsonify({'success': False, 'error': f'Invalid file type: {file.filename}. Only JPG, JPEG, PNG, JFIF allowed.'}), 400

        try:
            # Process in batches
            all_results = []
            current_batch = []
            
            for file in files:
                filename = secure_filename(f"{datetime.now().timestamp()}_{file.filename}")
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                image_bytes = file.read()
                current_batch.append((image_bytes, filename))
                
                if len(current_batch) >= batch_size:
                    batch_results = process_image_batch(current_batch)
                    all_results.extend(batch_results)
                    cleanup_batch_resources(batch_results)
                    current_batch = []
            
            # Process remaining items in last batch
            if current_batch:
                batch_results = process_image_batch(current_batch)
                all_results.extend(batch_results)
                cleanup_batch_resources(batch_results)
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

    # Handle image via URL link (single or multiple)
    elif 'link' in request.form:
        links = request.form.getlist('link')
        if not links or all(not link for link in links):
            return jsonify({'success': False, 'error': 'No image links provided'}), 400

        try:
            # Process in batches
            all_results = []
            current_batch = []
            
            for link in links:
                current_batch.append(link)
                
                if len(current_batch) >= batch_size:
                    batch_results = process_url_batch(current_batch)
                    all_results.extend(batch_results)
                    cleanup_batch_resources(batch_results)
                    current_batch = []
            
            # Process remaining items in last batch
            if current_batch:
                batch_results = process_url_batch(current_batch)
                all_results.extend(batch_results)
                cleanup_batch_resources(batch_results)

            # Prepare response
            successful_results = [r for r in all_results if 'error' not in r]
            failed_results = [r for r in all_results if 'error' in r]
            
            # Save successful results to MongoDB in bulk
            if successful_results:
                try:
                    db = client.get_database('endoskin')
                    analysis_collection = db.analysis_results
                    
                    bulk_operations = [
                        {
                            'originalLink': result['link'],
                            'savedPath': os.path.join(app.config['UPLOAD_FOLDER'], result['filename']),
                            'predictedCondition': result['condition'],
                            'confidence': result['confidence'],
                            'type': 'link',
                            'status': 'success',
                            'createdAt': datetime.now()
                        }
                        for result in successful_results
                    ]
                    
                    # Perform bulk insert
                    result = analysis_collection.insert_many(bulk_operations)
                    
                    # Update status for all successful results
                    for i, result in enumerate(successful_results):
                        result['status'] = 'success' if result.inserted_ids[i] else 'failed_to_save'
                        
                except Exception as e:
                    print(f"Error during bulk save to MongoDB: {e}")
                    for result in successful_results:
                        result['status'] = 'failed_to_save'

            # Format response
            response_data = {
                'success': True,
                'processed_count': len(all_results),
                'successful': len(successful_results),
                'failed': len(failed_results),
                'results': [{
                    'link': r['link'],
                    'success': 'error' not in r,
                    'prediction': {
                        'condition': r.get('condition'),
                        'confidence': f"{r.get('confidence', 0)}%" if 'condition' in r else None,
                        'timestamp': datetime.now().isoformat()
                    },
                    'imageUrl': f"/uploads/{r.get('filename')}" if 'filename' in r else None,
                    'error': r.get('error')
                } for r in all_results]
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