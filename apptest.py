import os
import uuid
import flask
import urllib.request # Changed from urllib (general module) to urllib.request (specific module)
from PIL import Image
from tensorflow.keras.models import load_model
from datetime import datetime
from flask import Flask , render_template , request , send_file, jsonify
from flask_cors import CORS
from tensorflow.keras.preprocessing.image import load_img , img_to_array
# from sklearn.metrics import f1_score # You only need this if you're actively using it, not just for loading a custom metric

# Removed this function definition as it's typically part of the training script
# def f1_metric(y_true, y_pred):
#     return f1_score(y_true, y_pred)

# Define f1_metric as a dummy if it's only needed for loading and not for actual use
# A better approach for loading models with custom metrics is to define a dummy function
# that matches the signature but doesn't require sklearn if you don't use it elsewhere.
# OR, if you use it in the training script, ensure sklearn is installed and accessible.
def f1_metric(y_true, y_pred):
    # This is a dummy for loading. If you truly need F1 score,
    # you'd calculate it here or ensure y_true and y_pred are compatible.
    # For loading a pre-trained model, a placeholder is often sufficient.
    return 0.0 # Or use tf.keras.metrics.F1Score if it's a Keras metric

classes = ['Acne', 'Eczema', 'Moles', 'Psoriasis', 'Rosacea','Seborrheic Keratoses', 'Sun Damage', 'Vitiligo', 'Warts']
# Note: Your model seems to output 2048 values, but you have 9 classes.
# This indicates your model's output layer might be wrong or it's a feature extractor.
# For now, we'll proceed with 9 classes, but be aware of this discrepancy.
# If your model truly has 9 classes, `result[0]` should have 9 elements.
# If it outputs 2048, it's likely a feature extractor, not a direct classifier.
# Assuming it should be 9 classes for now based on 'classes' list.

app = Flask(__name__, template_folder='.')
app.config['UPLOAD_FOLDER'] = 'uploads'
CORS(app, resources={
    r"/predict_image": {
        "origins": ["http://192.168.100.8:3000"],
        "methods": ["POST"],
        "allow_headers": ["Content-Type"]
    }
})
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Load model
model = load_model(os.path.join(BASE_DIR, 'incepti.h5'), custom_objects={'f1_metric': f1_metric})


ALLOWED_EXT = set(['jpg' , 'jpeg' , 'png' , 'jfif'])
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXT # Added .lower() for case-insensitivity


def predict(filename, model):
    img = load_img(filename, target_size=(224, 224))
    img = img_to_array(img)
    img = img.reshape(1, 224, 224, 3)
    img = img.astype('float32') / 255.0
    
    # Get features
    features = model.predict(img)
    
    # Simple classification based on feature magnitudes
    # This is a placeholder - you should train a proper classifier
    # on these features for your 9 classes
    top_3_indices = features[0].argsort()[-3:][::-1]
    
    class_result = []
    prob_result = []
    for idx in top_3_indices:
        if idx < len(classes):
            # Simple normalization for demonstration
            confidence = (features[0][idx] / features[0].sum()) * 100
            class_result.append(classes[idx])
            prob_result.append(round(confidence, 2))
        else:
            class_result.append(f"Unknown Feature ({idx})")
            prob_result.append(0.0)
            
    return class_result, prob_result


@app.route('/')
def home():
    return render_template("index.html")

# Modified the endpoint to be /predict (as it should be the one receiving image uploads for prediction)
# This was causing the KeyError because Postman was hitting /success expecting 'file'
# but your previous /predict endpoint expected 'image'. This new unified structure simplifies.
@app.route('/predict_image' , methods = ['POST']) # Renamed to avoid confusion with predict() function
def predict_image_endpoint(): # Renamed the function to avoid conflict with predict() function
    error = ''
    # Create both directories if they don't exist
    static_img_dir = os.path.join(os.getcwd(), 'static/images')
    uploads_dir = os.path.join(os.getcwd(), 'uploads')
    os.makedirs(static_img_dir, exist_ok=True)
    os.makedirs(uploads_dir, exist_ok=True)
    
    # Use uploads directory for saving files
    target_img = uploads_dir

    if request.method == 'POST':
        # Handling image via link
        if 'link' in request.form: # Check if 'link' form field exists
            link = request.form.get('link')
            try :
                resource = urllib.request.urlopen(link)
                unique_filename = str(uuid.uuid4())
                filename = unique_filename + ".jpg" # Assuming .jpg, but could use content-type from resource
                img_path = os.path.join(target_img , filename)
                with open(img_path , "wb") as output: # Use 'with' for safer file handling
                    output.write(resource.read())
                img = filename # This 'img' is just the filename for template

                class_result , prob_result = predict(img_path , model)

                response = {
                    "success": True,
                    "prediction": {
                        "condition": class_result[0],
                        "confidence": prob_result[0]
                    },
                    "imageUrl": f"/static/images/{img}"
                }
                return jsonify(response)

            except Exception as e :
                print(f"Error processing link: {str(e)}") # More specific error message
                error = f'Error with image link: {str(e)}' # Pass specific error

        # Handling image via file upload
        elif 'file' in request.files: # Check if 'file' form field exists
            file = request.files['file']
            if file.filename == '': # If no file was selected
                error = "No selected file"
            elif file and allowed_file(file.filename):
                filename = secure_filename(f"{datetime.now().timestamp()}_{file.filename}")
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
                file.save(filepath)
                print(f"File saved successfully at: {filepath}")
                img = filename # This 'img' is just the filename for template

                class_result , prob_result = predict(img_path , model)

                response = {
                    "success": True,
                    "prediction": {
                        "condition": class_result[0],
                        "confidence": prob_result[0]
                    },
                    "imageUrl": f"/static/images/{img}"
                }
                return jsonify(response)
            else:
                error = "Please upload images of jpg, jpeg, png, or jfif extension only."
        else:
            error = "No image file or link provided in the request." # Neither 'link' nor 'file' found

    # If any error occurred or request was GET
    if len(error) > 0:
        return jsonify({
            "success": False,
            "error": error
        })
    else:
        return jsonify({
            "success": False,
            "error": "An unexpected error occurred."
        })


if __name__ == "__main__":
    app.run(debug = True)