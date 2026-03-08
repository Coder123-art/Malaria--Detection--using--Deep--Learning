import os
import io
import json
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
import numpy as np
import base64
import matplotlib.pyplot as plt
global training_history, model

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = 'uploads'

# Create uploads folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global error handler - always return JSON
@app.errorhandler(Exception)
def handle_error(error):
    print(f"Error: {error}")
    import traceback
    traceback.print_exc()
    return jsonify({
        'success': False,
        'error': str(error)
    }), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'success': False,
        'error': 'Endpoint not found'
    }), 404

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
BASE_DIR = r"C:\Users\UTKARSH\.cache\kagglehub\datasets\iarunava\cell-images-for-detecting-malaria\versions\1\cell_images"
CATEGORIES = ['Parasitized', 'Uninfected']
MODEL_PATH = 'malaria_detector.keras'

# Global variables
training_history = None
model = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_model_from_disk():
    """Load the trained model from disk"""
    global model
    if os.path.exists(MODEL_PATH):
        try:
            import tensorflow as tf
            model = tf.keras.models.load_model(MODEL_PATH)
            print(f"Model loaded from {MODEL_PATH}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    return False

@app.route('/')
def home():
    """Home page"""
    model_exists = os.path.exists(MODEL_PATH)
    return render_template('index.html', model_exists=model_exists)
    

@app.route('/api/dataset-stats')
def dataset_stats():
    """Get dataset statistics"""
    try:
        print(f"Dataset path: {BASE_DIR}")
        print(f"Dataset exists: {os.path.exists(BASE_DIR)}")
        
        if not os.path.exists(BASE_DIR):
            print(f"ERROR: Dataset directory does not exist at {BASE_DIR}")
            return jsonify({
                'success': False,
                'error': f'Dataset directory not found at {BASE_DIR}',
                'path_checked': BASE_DIR
            }), 400
        
        counts = []
        for category in CATEGORIES:
            path = os.path.join(BASE_DIR, category)
            print(f"Checking category path: {path}")
            if os.path.exists(path):
                # Only count image files
                files = [f for f in os.listdir(path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                print(f"  - {category}: {len(files)} images")
                counts.append(len(files))
            else:
                print(f"  - {category}: path not found")
                counts.append(0)
        
        print(f"Counts: {counts}, Total: {sum(counts)}")
        
        return jsonify({
            'success': True,
            'categories': CATEGORIES,
            'counts': counts,
            'total': sum(counts)
        })
    except Exception as e:
        print(f"Error in dataset_stats: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': f'Error reading dataset: {str(e)}'
        }), 400

@app.route('/api/visualize-dataset')
def visualize_dataset():
    """Generate and return dataset distribution chart"""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        counts = []
        for category in CATEGORIES:
            path = os.path.join(BASE_DIR, category)
            if os.path.exists(path):
                files = [f for f in os.listdir(path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                counts.append(len(files))
            else:
                counts.append(0)
        
        plt.figure(figsize=(8, 5))
        sns.barplot(x=['Infected (Parasitized)', 'Healthy (Uninfected)'], y=counts, palette='viridis')
        plt.title("Malaria Dataset Distribution")
        plt.ylabel('Number of Images')
        plt.xlabel('Category')
        
        # Convert plot to base64 image
        img = io.BytesIO()
        plt.savefig(img, format='png', bbox_inches='tight')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close()
        
        return jsonify({
            'success': True,
            'image': f'data:image/png;base64,{plot_url}'
        })
    except Exception as e:
        print(f"Error in visualize_dataset: {e}")
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/train-model', methods=['POST'])
def train_model_endpoint():
    """Train the model and return progress"""
    try:
        global training_history, model
        
        # Import TensorFlow here to avoid slow startup
        import tensorflow as tf

        from tensorflow.keras.preprocessing.image import ImageDataGenerator
        from tensorflow.keras.applications import MobileNetV2
        from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
        from tensorflow.keras.models import Model
        
        # Clean dataset first
        print("--- Cleaning Dataset ---")
        for category in CATEGORIES:
            path = os.path.join(BASE_DIR, category)
            if os.path.exists(path):
                for file in os.listdir(path):
                    if not file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        try:
                            os.remove(os.path.join(path, file))
                        except:
                            pass
        
        # Data preprocessing
        print("--- Preprocessing Data ---")
        datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
        
        train_gen = datagen.flow_from_directory(
            BASE_DIR,
            target_size=(224, 224),
            batch_size=32,
            class_mode='binary',
            subset='training'
        )
        
        val_gen = datagen.flow_from_directory(
            BASE_DIR,
            target_size=(224, 224),
            batch_size=32,
            class_mode='binary',
            subset='validation'
        )
        
        # Build model
        print("--- Building Model ---")
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        base_model.trainable = False
        
        x = GlobalAveragePooling2D()(base_model.output)
        predictions = Dense(1, activation='sigmoid')(x)
        
        model = Model(inputs=base_model.input, outputs=predictions)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        # Train model
        print("--- Training Model ---")
        epochs = request.json.get('epochs', 3) if request.is_json else 3
        
        training_history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=epochs,
            verbose=1
        )
        
        # Save model
        model.save(MODEL_PATH)
        print("Model saved successfully")
        
        return jsonify({
            'success': True,
            'message': 'Model trained and saved successfully!',
            'epochs': epochs,
            'final_accuracy': float(training_history.history['accuracy'][-1]),
            'final_val_accuracy': float(training_history.history['val_accuracy'][-1])
        })
    except Exception as e:
        print(f"Error in train_model: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/training-plot')
def training_plot():
    """Generate and return training history plot"""
    try:
        if training_history is None:
            return jsonify({'success': False, 'error': 'No training history available'}), 400
        
        plt.figure(figsize=(10, 6))
        plt.plot(training_history.history['accuracy'], label='Training Accuracy', color='blue', marker='o')
        plt.plot(training_history.history['val_accuracy'], label='Validation Accuracy', color='orange', marker='o')
        plt.title('AI Learning Progress')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch (Rounds)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Convert plot to base64 image
        img = io.BytesIO()
        plt.savefig(img, format='png', bbox_inches='tight')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close()
        
        return jsonify({
            'success': True,
            'image': f'data:image/png;base64,{plot_url}'
        })
    except Exception as e:
        print(f"Error in training_plot: {e}")
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/predict', methods=['POST'])
def predict():
    """Make prediction on uploaded image"""
    try:
        if not os.path.exists(MODEL_PATH):
            return jsonify({'success': False, 'error': 'Model not trained yet. Please train the model first.'}), 400
        
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'success': False, 'error': 'File type not allowed. Use PNG, JPG, or JPEG.'}), 400
        
        # Import TensorFlow here
        import tensorflow as tf
        from tensorflow.keras.preprocessing import image
        
        # Load model
        global model
        if model is None:
            load_model_from_disk()
            if model is None:
                return jsonify({'success': False, 'error': 'Failed to load model'}), 400
        
        # Read and preprocess image
        img_data = file.read()
        img = image.load_img(io.BytesIO(img_data), target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        
        # Make prediction
        prediction = model.predict(img_array, verbose=0)
        confidence = float(prediction[0][0])
        
        # Interpret result
        if confidence > 0.5:
            result = "Healthy (Uninfected)"
            confidence_percent = (confidence * 100)
        else:
            result = "Infected (Parasitized)"
            confidence_percent = ((1 - confidence) * 100)
        
        return jsonify({
            'success': True,
            'prediction': result,
            'confidence': round(confidence_percent, 2),
            'raw_score': round(confidence, 4)
        })
    except Exception as e:
        print(f"Error in predict: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/model-info')
def model_info():
    """Get model information"""
    try:
        if not os.path.exists(MODEL_PATH):
            return jsonify({'exists': False})
        
        model_size = os.path.getsize(MODEL_PATH) / (1024 * 1024)
        return jsonify({
            'exists': True,
            'path': MODEL_PATH,
            'size_mb': round(model_size, 2)
        })
    except Exception as e:
        print(f"Error in model_info: {e}")
        return jsonify({'success': False, 'error': str(e)}), 400

if __name__ == '__main__':
    print("=" * 60)
    print("Starting Malaria Detection Flask App")
    print("=" * 60)
    print("Open your browser and go to http://localhost:5000")
    print("=" * 60)
    app.run(debug=True, port=5000, use_reloader=False)
