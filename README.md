🧠 Speech-Based Parkinson’s Disease Detection using Deep Learning

An end-to-end machine learning pipeline that detects Parkinson’s Disease from speech signals using advanced audio feature extraction and a neural network model.

🚀 Overview

This project leverages speech signal processing and deep learning to identify Parkinson’s disease based on vocal biomarkers. The system extracts meaningful features from raw audio and classifies them using a trained neural network.

📊 Dataset
Total samples: 3400 audio files
Classes:
Healthy: 1700
Parkinson’s Disease (PD): 1700
Balanced dataset across multiple vowel sounds
⚙️ Pipeline
Data Loading
Organized dataset by vowels and conditions (Healthy / PD)
Preprocessing
Resampling to 16 kHz
Normalization
Fixed audio duration (3 seconds)
Feature Extraction (26 Features)
Statistical features (mean, variance, skewness, kurtosis, RMS, etc.)
Wavelet features + entropy
MFCC features (mean & std)
Spectral features (centroid, ZCR)
Signal entropy
Data Splitting
Train: 2380
Validation: 340
Test: 680
Model
Fully connected neural network (PyTorch)
Architecture: 128 → 64 → 2
Dropout regularization
Evaluation
Accuracy, Precision, Recall, F1-score
ROC-AUC, MCC, Cohen’s Kappa
Confusion matrix visualization
📈 Results
Validation Accuracy: 88.24%
Test Accuracy: ~87.94%
Confusion Matrix
               Predicted
               Healthy   PD
Actual Healthy   304     36
Actual PD         46     294
🛠️ Tech Stack
Languages: Python
Libraries:
PyTorch
Librosa
NumPy, SciPy
Scikit-learn
PyWavelets
Matplotlib, Seaborn
📁 Project Structure
├── data/                    # Dataset (not included)
├── pd_optimized_results/
│   ├── graphs/
│   ├── models/
│   └── results.pkl
├── main.py                 # Main pipeline script
└── README.md
▶️ How to Run
# Clone the repository
git clone https://github.com/your-username/parkinsons-detection.git

# Navigate to project
cd parkinsons-detection

# Install dependencies
pip install -r requirements.txt

# Run the pipeline
python main.py
📌 Key Features
End-to-end ML pipeline from raw audio to prediction
Efficient feature extraction (26 features per sample)
Reproducible results with fixed random seeds
Balanced dataset handling
Detailed performance evaluation
🔮 Future Improvements
Use Transformer / CNN models for better accuracy
Real-time prediction system
Deployment as a web/mobile application
Use larger and more diverse datasets
