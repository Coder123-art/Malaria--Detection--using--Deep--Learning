# Malaria Detection System - Flask Web Application

A deep learning-based web application for detecting malaria from cell microscopy images using TensorFlow and Flask.

## Features

- **Dataset Visualization**: View statistics and distribution of the training dataset
- **Model Training**: Train a MobileNetV2-based neural network with customizable epochs
- **Training Visualization**: Plot accuracy and validation curves
- **Image Prediction**: Upload cell images to get predictions (Infected/Healthy)
- **Web Interface**: User-friendly Flask web application

## Prerequisites

- Python 3.8+
- CUDA/GPU support (optional, but recommended for faster training)
- Malaria cell image dataset from Kaggle

## Installation

### 1. Create and Activate Virtual Environment

```powershell
# Create virtual environment
python -m venv .venv

# Activate it
.\.venv\Scripts\Activate.ps1
```

### 2. Install Dependencies

```powershell
pip install -r requirements.txt
```

Or install manually:
```powershell
pip install flask tensorflow numpy matplotlib seaborn werkzeug
```

## Running the Application

### 1. Start the Flask Server

```powershell
.\.venv\Scripts\python.exe app.py
```

### 2. Open in Browser

Navigate to: **http://localhost:5000**

## Usage

### Step 1: Load Dataset Statistics
- Click "Load Dataset Stats" to see how many training images are available
- The system will display the distribution of infected vs. healthy cells

### Step 2: Train the Model
1. Set the number of epochs (recommended: 3-10)
2. Click "Start Training"
3. Wait for training to complete (takes a few minutes)
4. The system will save the trained model as `malaria_detector.keras`

### Step 3: View Training Results
- Click "View Training Plot" to see how the model's accuracy improved
- Compare training accuracy vs. validation accuracy

### Step 4: Make Predictions
1. Upload a cell image (PNG, JPG, or JPEG)
2. Click "Make Prediction"
3. Get instant results with confidence percentage

## API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/` | GET | Main web interface |
| `/api/dataset-stats` | GET | Get dataset statistics |
| `/api/visualize-dataset` | GET | Get dataset visualization chart |
| `/api/train-model` | POST | Train the model |
| `/api/training-plot` | GET | Get training history plot |
| `/api/predict` | POST | Make prediction on image |
| `/api/model-info` | GET | Get model information |

## Dataset Structure

```
C:\Users\UTKARSH\.cache\kagglehub\datasets\iarunava\cell-images-for-detecting-malaria\versions\1\cell_images\
├── Parasitized/
│   ├── image1.png
│   ├── image2.png
│   └── ...
└── Uninfected/
    ├── image1.png
    ├── image2.png
    └── ...
```

## Configuration

Edit `app.py` to change:
- **BASE_DIR**: Path to your dataset
- **MODEL_PATH**: Where to save/load the model
- **PORT**: Flask server port (default: 5000)
- **MAX_CONTENT_LENGTH**: Maximum file upload size

## Model Details

- **Base Model**: MobileNetV2 (pre-trained on ImageNet)
- **Input Size**: 224x224 pixels
- **Batch Size**: 32
- **Image Formats**: PNG, JPG, JPEG

## Troubleshooting

### Issue: "Model not trained yet"
- Click "Start Training" first to create the model

### Issue: Dataset not found
- Update `BASE_DIR` in `app.py` to point to your dataset location
- Run: `python -c "import os; print(os.path.exists('C:\\Users\\UTKARSH\\.cache\\kagglehub\\...'))"`

### Issue: Port already in use
- Change port in `app.py`: `app.run(debug=True, port=5001)`

### Issue: Out of memory during training
- Reduce batch_size in `app.py`
- Use fewer epochs
- Use a GPU if available

## Project Structure

```
Malaria--Detection--using--Deep--Learning/
├── app.py                    # Flask application
├── train_model.py            # Original training script
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── malaria_detector.keras    # Trained model (auto-generated)
├── templates/
│   └── index.html           # Web interface
└── uploads/                 # Temporary uploaded images
```

## Performance

- Training time: ~5-10 minutes per epoch (depends on GPU)
- Model accuracy: ~95-98% on validation set
- Prediction time: ~500ms per image

## License

Educational use - Kaggle dataset https://www.kaggle.com/iarunava/cell-images-for-detecting-malaria

## References

- MobileNetV2: https://arxiv.org/abs/1801.04381
- TensorFlow: https://www.tensorflow.org/
- Kaggle Dataset: https://www.kaggle.com/iarunava/cell-images-for-detecting-malaria
