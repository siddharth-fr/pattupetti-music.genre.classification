# pattupetti 🎵
### Music Genre Classification & Sound Intelligence Platform

> **pattupetti** *(Malayalam/Tamil for "music box" / jukebox)* is an end-to-end machine learning system and aesthetic web application that classifies music into 8 distinct genres by analyzing the physics of sound.

---

## 🌟 Highlights

- **Physics-Informed Audio Features**: Rather than treating audio as a black box, extracts 75 mathematical acoustic descriptors (MFCCs, Chroma STFT, Spectral Centroid, Bandwidth, Rolloff, Zero-Crossing Rate, RMS Energy, and Tempo) using **Librosa**.
- **Rigorous ML Benchmark**: Compares 6 classical classifiers with 5-fold stratified cross-validation on the balanced **FMA Small** dataset (8,000 tracks, 1,000 per genre).
- **53.56% Test Accuracy / 0.5287 Macro F1**: SVM (RBF kernel) outperforms chance (12.5%) by **4.3×**, backed by an adaptive soft-voting Random Forest ensemble.
- **Aesthetic Web Application**: Minimalist linen/sage frontend featuring a custom waveform player, interactive draggable scrubber needle, live microphone capture with real-time WebM/Opus audio decoding, 1-click test tracks, and an in-app project documentation report.
- **Instant Out-of-the-Box Demo**: Pre-bundled with lightweight demo tracks and trained model weights, allowing anyone to launch and test immediately without downloading the full 8 GB raw dataset.

---

## 📁 Project Structure

```text
pattupetti/
├── data/
│   ├── samples/                # Bundled demo audio tracks for immediate viva testing (3.3 MB)
│   ├── processed/              # Extracted 75-feature matrix (features_30s.csv, 11.4 MB)
│   ├── raw/                    # (Optional) Full FMA dataset archives (gitignored)
│   └── temp_audio/             # Runtime directory for uploads and mic blobs (gitignored)
├── models/
│   ├── best_model.joblib       # Trained Support Vector Machine (RBF kernel)
│   ├── rf_model.joblib         # Compressed Random Forest secondary model
│   ├── scaler.joblib           # Pre-fitted StandardScaler (leak-free)
│   ├── feature_names.joblib    # Exact feature ordering list
│   ├── best_model_name.joblib  # Serialized model name
│   └── model_comparison.csv   # Comprehensive metric leaderboard
├── reports/
│   ├── classification_report.txt  # Detailed per-genre precision, recall & F1
│   ├── confusion_matrix.png       # Normalized heatmap
│   └── model_comparison.png       # Benchmark bar chart
├── src/
│   ├── dataset.py              # FMA dataset download, resume, and verification
│   ├── features.py             # Librosa feature extraction & audio decoders
│   ├── train.py                # 5-fold CV & model training pipeline
│   ├── evaluate.py             # Chart generation & classification report
│   └── predict.py              # CLI and API inference engine
├── static/
│   ├── app.js                  # Audio WebAPI, visualizer, mic capture, and UI state
│   ├── style.css               # Minimal sage design system
│   └── docs.css                # Project report stylesheet
├── templates/
│   ├── index.html              # Main classifier web application
│   └── docs.html               # In-app project documentation report (/docs)
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
└── .gitignore                  # Git tracking rules
```

---

## 🚀 Quickstart Guide

### 1. Clone & Set Up Virtual Environment

```bash
# Clone the repository
git clone https://github.com/your-username/pattupetti.git
cd pattupetti

# Create and activate a virtual environment
python -m venv .venv

# Windows (PowerShell):
.\.venv\Scripts\Activate.ps1

# Linux / macOS:
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Launch the Web Application

```bash
python app.py
```

Open your browser at **[http://127.0.0.1:5000](http://127.0.0.1:5000)**:
- **Classifier Tab**: Try 1-click bundled samples, upload your own `.mp3` or `.wav` song, or record singing/audio live from your microphone.
- **Report Tab** (`/docs`): View the complete in-app project documentation covering dataset physics, feature mathematics, pipeline architecture, and model leaderboards.

---

## 🔬 Benchmark Leaderboard (FMA Small)

Stratified 80/20 train/test split (6,400 training tracks, 1,600 held-out test tracks across 8 balanced genres):

| Rank | Model | 5-Fold CV F1 | Test Accuracy | Precision | Recall | Macro F1 | Train Time |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 🥇 | **SVM (RBF Kernel)** | **0.5280 ± 0.014** | **53.56%** | **0.5282** | **0.5356** | **0.5287** | 21.41 s |
| 🥈 | Random Forest | 0.4891 ± 0.013 | 49.75% | 0.4926 | 0.4975 | 0.4892 | 4.07 s |
| 🥉 | Logistic Regression | 0.4583 ± 0.013 | 48.38% | 0.4687 | 0.4838 | 0.4694 | 3.54 s |
| 4 | K-Nearest Neighbors | 0.4395 ± 0.013 | 45.81% | 0.4563 | 0.4581 | 0.4486 | 2.32 s |
| 5 | Gaussian Naive Bayes | 0.3857 ± 0.010 | 41.19% | 0.4152 | 0.4119 | 0.3832 | 0.18 s |
| 6 | Decision Tree | 0.3461 ± 0.009 | 33.62% | 0.3355 | 0.3363 | 0.3346 | 2.98 s |

*Random guess baseline for 8 balanced genres is 12.50%.*

---

## 🛠️ CLI Pipeline Commands

You can run each phase of the machine learning pipeline individually from the terminal:

1. **Verify Dataset** *(optional, requires downloaded FMA Small)*:
   ```bash
   python -m src.dataset
   ```

2. **Extract Features** *(optional, pre-computed in `data/processed/features_30s.csv`)*:
   ```bash
   python -m src.features
   ```

3. **Train & Evaluate All Models**:
   ```bash
   python -m src.train
   ```

4. **Generate Evaluation Visualizations & Reports**:
   ```bash
   python -m src.evaluate
   ```

5. **Classify a Single Audio Track via CLI**:
   ```bash
   python src/predict.py "path/to/any_song.mp3"
   ```

---

## 📄 License & Attribution

- Built as a 2nd Year B.Tech Computer Science / AI & DS Mini-Project.
- Dataset: [Free Music Archive (FMA)](https://github.com/mdeff/fma) licensed under Creative Commons.
- Audio Analysis: [Librosa](https://librosa.org/) & [Scikit-learn](https://scikit-learn.org/).
