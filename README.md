# 🦠 Malaria Detection Using Deep Learning

<div align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/TensorFlow-2.13+-orange.svg" alt="TensorFlow Version">
  <img src="https://img.shields.io/badge/Flask-2.3+-lightgrey.svg" alt="Flask Version">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
</div>

## 📋 Overview

Malaria is a life-threatening disease caused by parasites transmitted through mosquito bites, affecting millions of people worldwide, particularly in tropical and subtropical regions. Traditional diagnosis relies on manual microscopic examination of blood smears by trained professionals, which is time-consuming, labor-intensive, and subject to human error.

This project implements an **automated malaria detection system** using **Convolutional Neural Networks (CNNs)** to analyze microscopic blood cell images. The system can distinguish between infected (parasitized) and healthy (uninfected) red blood cells with high accuracy, providing a fast, reliable, and scalable diagnostic tool.

## 🎯 Key Features

### 🔬 **Advanced AI Technology**
- **MobileNetV2 Architecture**: Pre-trained on ImageNet, fine-tuned for malaria detection
- **Transfer Learning**: Leverages existing knowledge for faster training and better performance
- **Binary Classification**: Distinguishes between parasitized and uninfected cells
- **Real-time Prediction**: Instant results with confidence scores

### 🌐 **Web Application**
- **Flask-based Web Interface**: User-friendly browser-based application
- **Interactive Dashboard**: Visualize dataset statistics and training progress
- **File Upload System**: Drag-and-drop image upload functionality
- **Training Visualization**: Real-time plots of model accuracy and loss curves

### 📊 **Comprehensive Analytics**
- **Dataset Visualization**: Bar charts showing class distribution
- **Training Metrics**: Accuracy and validation curves over epochs
- **Model Performance**: Detailed evaluation metrics and statistics
- **Prediction Confidence**: Percentage-based confidence scores

### 🛠 **Developer-Friendly**
- **Modular Architecture**: Clean separation of concerns
- **RESTful API**: Well-documented endpoints for integration
- **Error Handling**: Comprehensive error reporting and logging
- **Scalable Design**: Easy to extend and modify

## 🏗️ Architecture

### **Model Architecture**
```
Input Image (224x224x3)
        │
    MobileNetV2 Base
    (Pre-trained on ImageNet)
        │
Global Average Pooling
        │
    Dense Layer (1 neuron)
    (Sigmoid Activation)
        │
    Binary Classification
    (0 = Parasitized, 1 = Uninfected)
```

### **System Architecture**
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Web Browser   │────│    Flask App    │────│   TensorFlow    │
│                 │    │                 │    │     Model       │
│ • Upload Images │    │ • API Endpoints │    │                 │
│ • View Results  │    │ • Data Processing│    │ • CNN Inference │
│ • Training UI   │    │ • Model Training │    │ • Predictions   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🚀 Quick Start

### **Prerequisites**
- Python 3.8 or higher
- pip package manager
- Git (optional, for cloning)

### **Installation**

#### 1. **Clone or Download the Repository**
```bash
git clone https://github.com/your-username/malaria-detection.git
cd malaria-detection
```

#### 2. **Create Virtual Environment**
```bash
# Windows
python -m venv .venv
.\.venv\Scripts\activate

# Linux/Mac
python -m venv .venv
source .venv/bin/activate
```

#### 3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

#### 4. **Download Dataset**
The project uses the [Malaria Cell Images Dataset](https://www.kaggle.com/iarunava/cell-images-for-detecting-malaria) from Kaggle.

**Option A: Manual Download**
- Download from Kaggle
- Extract to: `C:\Users\[USERNAME]\.cache\kagglehub\datasets\iarunava\cell-images-for-detecting-malaria\versions\1\cell_images\`

**Option B: Use Kaggle API**
```bash
pip install kaggle
kaggle datasets download iarunava/cell-images-for-detecting-malaria
```

### **Running the Application**

#### **Start the Web Server**
```bash
python app.py
```

#### **Access the Application**
Open your browser and navigate to: **http://localhost:5000**

## 📖 Usage Guide

### **Step 1: Dataset Exploration**
1. Click **"Load Dataset Stats"** to view dataset information
2. Review the distribution chart showing infected vs. healthy cells
3. Check total image counts for each category

### **Step 2: Model Training**
1. Set the number of training epochs (recommended: 3-10)
2. Click **"Start Training"** to begin the training process
3. Monitor the training progress in the terminal/console
4. Wait for training completion (may take 5-15 minutes)

### **Step 3: Training Analysis**
1. Click **"View Training Plot"** after training completes
2. Analyze the accuracy curves for training and validation
3. Review final model performance metrics

### **Step 4: Making Predictions**
1. Click **"Choose Image"** to select a cell image
2. Preview the uploaded image
3. Click **"Make Prediction"** to get results
4. View the prediction result and confidence score

## 🔧 API Reference

### **Endpoints**

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Main web interface |
| `GET` | `/api/dataset-stats` | Get dataset statistics |
| `GET` | `/api/visualize-dataset` | Generate dataset distribution chart |
| `POST` | `/api/train-model` | Train the machine learning model |
| `GET` | `/api/training-plot` | Get training history visualization |
| `POST` | `/api/predict` | Make prediction on uploaded image |
| `GET` | `/api/model-info` | Get model information |

### **API Examples**

#### **Get Dataset Statistics**
```bash
curl http://localhost:5000/api/dataset-stats
```

#### **Train Model**
```bash
curl -X POST http://localhost:5000/api/train-model \
  -H "Content-Type: application/json" \
  -d '{"epochs": 5}'
```

#### **Make Prediction**
```bash
curl -X POST http://localhost:5000/api/predict \
  -F "file=@cell_image.png"
```

## 📊 Dataset Information

### **Source**
- **Dataset**: Malaria Cell Images Dataset
- **Provider**: Kaggle (iarunava)
- **Size**: ~27,000+ images
- **Classes**: 2 (Parasitized, Uninfected)

### **Data Structure**
```
dataset/
├── cell_images/
│   ├── Parasitized/
│   │   ├── image1.png
│   │   ├── image2.png
│   │   └── ...
│   └── Uninfected/
│       ├── image1.png
│       ├── image2.png
│       └── ...
```

### **Preprocessing**
- **Image Size**: 224x224 pixels (MobileNetV2 standard)
- **Color Mode**: RGB
- **Normalization**: Pixel values scaled to [0,1]
- **Split**: 80% training, 20% validation

## 🧠 Model Details

### **Architecture Specifications**
- **Base Model**: MobileNetV2
- **Input Shape**: (224, 224, 3)
- **Output**: Single sigmoid neuron
- **Loss Function**: Binary Crossentropy
- **Optimizer**: Adam
- **Metrics**: Accuracy

### **Training Parameters**
- **Batch Size**: 32
- **Learning Rate**: Adam default (0.001)
- **Epochs**: Configurable (default: 3)
- **Validation Split**: 20%

### **Performance Metrics**
- **Accuracy**: >95% (typical)
- **Precision**: High for both classes
- **Recall**: Balanced detection
- **F1-Score**: >0.95

## 🛠️ Configuration

### **Environment Variables**
```bash
# TensorFlow Configuration
TF_CPP_MIN_LOG_LEVEL=3          # Reduce TensorFlow logging
TF_ENABLE_ONEDNN_OPTS=1         # Enable oneDNN optimizations

# Flask Configuration
FLASK_ENV=development           # Development mode
FLASK_DEBUG=1                   # Enable debug mode
```

### **Model Hyperparameters**
Edit `app.py` to modify:
```python
# Training parameters
EPOCHS = 3                      # Number of training epochs
BATCH_SIZE = 32                 # Batch size for training
LEARNING_RATE = 0.001           # Adam learning rate

# Model architecture
INPUT_SHAPE = (224, 224, 3)    # Input image dimensions
```

## 🔍 Troubleshooting

### **Common Issues**

#### **TensorFlow Import Errors**
```bash
# Solution: Reinstall TensorFlow
pip uninstall tensorflow
pip install tensorflow-cpu==2.13.0
```

#### **Dataset Not Found**
```bash
# Ensure dataset is in correct location
C:\Users\[USERNAME]\.cache\kagglehub\datasets\iarunava\cell-images-for-detecting-malaria\versions\1\cell_images\
```

#### **Port Already in Use**
```python
# Change port in app.py
app.run(debug=True, port=5001)
```

#### **Memory Issues**
```python
# Reduce batch size
BATCH_SIZE = 16  # Instead of 32
```

#### **Slow Training**
- Use GPU if available
- Reduce epochs for testing
- Use smaller batch size

### **Debug Mode**
```bash
# Run with detailed logging
python app.py
```

## 📈 Performance Optimization

### **GPU Acceleration**
```bash
# Install GPU version
pip install tensorflow-gpu
```

### **Model Optimization**
- **Quantization**: Reduce model size
- **Pruning**: Remove unnecessary weights
- **Knowledge Distillation**: Smaller student models

### **Inference Optimization**
- **Batch Processing**: Process multiple images
- **Model Caching**: Keep model in memory
- **Async Processing**: Non-blocking predictions

## 🤝 Contributing

### **Development Setup**
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

### **Code Style**
- Follow PEP 8 guidelines
- Use descriptive variable names
- Add docstrings to functions
- Include type hints where possible

### **Testing**
```bash
# Run basic tests
python -m pytest tests/

# Test API endpoints
python test_api.py
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Dataset**: Malaria Cell Images Dataset by iarunava on Kaggle
- **Framework**: TensorFlow and Keras
- **Web Framework**: Flask
- **Visualization**: Matplotlib and Seaborn

## 📞 Support

For questions, issues, or contributions:
- **GitHub Issues**: Report bugs and request features
- **Pull Requests**: Submit improvements
- **Email**: Contact the maintainers

## 🔄 Version History

### **v1.0.0** (Current)
- Initial release with Flask web interface
- MobileNetV2-based malaria detection
- Dataset visualization and training plots
- RESTful API for integration

### **Future Enhancements**
- [ ] Multi-class classification (different parasite species)
- [ ] Real-time video analysis
- [ ] Mobile app companion
- [ ] Cloud deployment options
- [ ] Advanced model architectures (EfficientNet, ResNet)

---

<div align="center">
  <p><strong>Made with ❤️ for global health</strong></p>
  <p>Contributing to malaria eradication through technology</p>
</div>
