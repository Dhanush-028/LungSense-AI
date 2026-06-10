# 🫁 LungSense AI — Respiratory Disease Classifier

A deep learning-powered web application that classifies lung sounds into respiratory disease categories using the ICBHI 2017 dataset.

## 🔗 Live Demo
[Click here to view the live app](https://lungsense-ai-production.up.railway.app)

### ICBHI 2017 Dataset
This project uses the **ICBHI 2017 Scientific Challenge Dataset**, which contains:
- 920 audio recordings from 126 patients
- Annotated lung sound cycles with 4 classes:
  - **Normal** — healthy lung sounds
  - **Crackle** — discontinuous, explosive sounds (pneumonia, fibrosis)
  - **Wheeze** — high-pitched musical sounds (asthma, COPD)
  - **Both** — presence of both crackles and wheezes

### How It Works
1. User uploads a `.wav` lung sound recording
2. Audio is converted to a **Mel spectrogram** (visual representation of sound)
3. The spectrogram is passed through a **CNN/MobileNetV2** deep learning model
4. A **Random Forest** baseline model also classifies independently
5. An **ensemble approach** combines both predictions for final classification
6. Results are displayed with risk scores and class probabilities

## 🛠️ Tech Stack
- **Backend:** Python, Flask
- **Deep Learning:** TensorFlow, Keras, MobileNetV2
- **Machine Learning:** Scikit-learn, Random Forest
- **Audio Processing:** Librosa
- **Database:** SQLite
- **Frontend:** HTML, CSS, JavaScript
- **Deployment:** Railway

## 📋 Features
- Upload WAV lung sound files
- Real-time classification into Normal, Wheeze, Crackle, Both
- Risk score and class probability display
- Clinical report generation
- Prediction history tracking
- Model comparison (CNN vs Random Forest vs Ensemble)
- Lung health exercise games
