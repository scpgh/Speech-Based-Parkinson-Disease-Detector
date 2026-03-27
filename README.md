# 🧠 Speech-Based Parkinson’s Disease Detection using Deep Learning

An end-to-end Deep learning pipeline that detects Parkinson’s Disease from speech signals using audio feature extraction and a neural network.

---

## 🚀 Overview

This project uses speech signal processing and deep learning to detect Parkinson’s disease from voice recordings. It extracts meaningful features from raw audio and classifies them using a trained neural network.

---

## 📊 Dataset

- Total samples: 1700 audio files
- Classes:
  - Healthy: 850
  - Parkinson’s Disease (PD): 850
- Balanced dataset across multiple vowel sounds

---

## ⚙️ Pipeline

1. Data Loading  
   - Organized dataset by vowels and conditions (Healthy / PD)

2. Preprocessing  
   - Resampling to 16 kHz  
   - Normalization  
   - Fixed audio duration (3 seconds)

3. Feature Extraction (26 Features)  
   - Statistical features (mean, variance, skewness, kurtosis, RMS, etc.)  
   - Wavelet features + entropy  
   - MFCC features (mean & std)  
   - Spectral features (centroid, ZCR)  
   - Signal entropy  

4. Data Splitting  
   - Train: 2380  
   - Validation: 340  
   - Test: 680  

5. Model  
   - Fully connected neural network (PyTorch)  
   - Architecture: 128 → 64 → 2  
   - Dropout regularization  

6. Evaluation  
   - Accuracy, Precision, Recall, F1-score  
   - ROC-AUC, MCC, Cohen’s Kappa  
   - Confusion matrix visualization  

---

## 📈 Results

- Validation Accuracy: 88.24%  
- Test Accuracy: ~87.94%  

Confusion Matrix:

               Predicted
               Healthy   PD
Actual Healthy   304     36
Actual PD         46     294

---

## 🛠️ Tech Stack

- Python  
- PyTorch  
- Librosa  
- NumPy, SciPy  
- Scikit-learn  
- PyWavelets  
- Matplotlib, Seaborn  

---

## 📁 Project Structure

parkinsons-detection/
│
├── data/                    # Dataset (not included)
├── pd_optimized_results/
│   ├── graphs/
│   ├── models/
│   └── results.pkl
├── main.py                 # Main pipeline script
└── README.md

---

## ▶️ How to Run

git clone https://github.com/scpgh/Speech-Based-Parkinson-Disease-Detector
cd parkinsons-detection
pip install -r requirements.txt
python main.py

---

## 📌 Key Features

- End-to-end ML pipeline from raw audio to prediction  
- Efficient feature extraction (26 features per sample)  
- Reproducible results with fixed random seeds  
- Balanced dataset handling  
- Detailed performance evaluation  

---

## 🔮 Future Improvements

- Use Transformer / CNN models for better accuracy  
- Real-time prediction system  
- Deployment as a web/mobile application  
- Use larger and more diverse datasets  

---

## 👨‍💻 Author

Srichaitanya Panda
GitHub: https://github.com/scpgh 

---

## ⭐ If you like this project

Give it a star on GitHub!
