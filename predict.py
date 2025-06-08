import tensorflow as tf
import numpy as np
import sys
import json
from PIL import Image

# Load model
model = tf.keras.models.load_model('incepti.h5')

# Class labels
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

def preprocess_image(image_path):
    # Load and preprocess image
    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict(image_path):
    try:
        # Preprocess and predict
        processed_img = preprocess_image(image_path)
        predictions = model.predict(processed_img)
        predicted_class = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class])
        
        return {
            'class': int(predicted_class),
            'confidence': confidence,
            'status': 'success'
        }
    except Exception as e:
        return {
            'status': 'error',
            'message': str(e)
        }

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(json.dumps({'status': 'error', 'message': 'Invalid arguments'}))
        sys.exit(1)
        
    image_path = sys.argv[1]
    result = predict(image_path)
    print(json.dumps(result))
