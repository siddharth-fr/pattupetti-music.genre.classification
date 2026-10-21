"""
Inference Script for Music Genre Classification

Loads the saved best model and scaler to predict the music genre of any audio file (.mp3 or .wav).

Usage:
    python src/predict.py "path/to/song.mp3"
"""

import sys
import argparse
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features import extract_features_from_audio

# In-memory artifact cache to avoid expensive disk reloads on every prediction request
_MODEL_CACHE = {}


def get_loaded_models(models_dir: str = "models"):
    """Loads and caches model artifacts in memory."""
    global _MODEL_CACHE
    if models_dir not in _MODEL_CACHE:
        models_path = Path(models_dir)
        if not models_path.exists():
            raise FileNotFoundError(f"Models directory not found: {models_dir}")
        model = joblib.load(models_path / "best_model.joblib")
        scaler = joblib.load(models_path / "scaler.joblib")
        feature_names = joblib.load(models_path / "feature_names.joblib")
        best_name = joblib.load(models_path / "best_model_name.joblib")
        rf_model_path = models_path / "rf_model.joblib"
        rf_model = joblib.load(rf_model_path) if rf_model_path.exists() else None
        _MODEL_CACHE[models_dir] = {
            "model": model,
            "scaler": scaler,
            "feature_names": feature_names,
            "best_name": best_name,
            "rf_model": rf_model,
        }
    return _MODEL_CACHE[models_dir]


def predict_genre(audio_path: str, models_dir: str = "models") -> dict:
    """
    Predicts the genre of an input audio file.
    
    Args:
        audio_path: Path to .mp3, .wav, .m4a, or .webm file.
        models_dir: Directory containing saved model artifacts.
        
    Returns:
        Dictionary with predicted genre and confidence probabilities.
    """
    audio_file = Path(audio_path)
    if not audio_file.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Load artifacts from cache
    artifacts = get_loaded_models(models_dir)
    model = artifacts["model"]
    scaler = artifacts["scaler"]
    feature_names = artifacts["feature_names"]
    best_name = artifacts["best_name"]
    rf_model = artifacts["rf_model"]

    print(f"\nAnalyzing track: {audio_file.name}")
    print(f"Loaded classifier: {best_name}")
    print("Extracting acoustic features (MFCC, Chroma, Spectral, RMS, Tempo)...")

    # Extract features (raising informative errors directly)
    raw_features = extract_features_from_audio(str(audio_file), raise_on_error=True)
    if raw_features is None:
        raise ValueError(f"Could not extract features from '{audio_path}'. File may be corrupt, unsupported, or silent.")

    # Align features exactly to training feature order
    feature_vector = [raw_features[col] for col in feature_names]
    X = np.array(feature_vector).reshape(1, -1)

    # Scale using the saved training scaler
    X_scaled = scaler.transform(X)

    # Robust Outlier Regularization:
    # High-dimensional RBF kernels (75 dimensions) decay exponentially to zero if any
    # feature has extreme z-score outliers (e.g. ambient mic noise or high MFCC variance).
    # Clamping to [-3.2, 3.2] preserves statistical ranking while keeping Euclidean distance bounded.
    X_scaled_regularized = np.clip(X_scaled, -3.2, 3.2)

    # Compute probabilities with primary and secondary models
    if hasattr(model, "predict_proba"):
        probs_svm = model.predict_proba(X_scaled_regularized)[0]
        classes = model.classes_

        # Detect RBF kernel underflow:
        # When an input is out of distribution, SVM RBF kernel evaluations all drop to ~0,
        # collapsing the decision function to the fixed intercept vector (~44.2% Experimental).
        exp_idx = list(classes).index("Experimental") if "Experimental" in classes else -1
        is_svm_collapsed = (exp_idx >= 0 and 0.435 < probs_svm[exp_idx] < 0.450 and np.max(probs_svm) == probs_svm[exp_idx])

        if rf_model is not None and hasattr(rf_model, "predict_proba"):
            probs_rf = rf_model.predict_proba(X_scaled_regularized)[0]
            if is_svm_collapsed:
                # Dynamically switch to Random Forest if SVM kernel underflowed
                final_probs = probs_rf
                active_model_name = "Random Forest (Adaptive Fallback)"
            else:
                # Soft Voting Ensemble: 60% SVM (best benchmark) + 40% RF (distribution regularizer)
                final_probs = 0.60 * probs_svm + 0.40 * probs_rf
                active_model_name = f"{best_name} + Random Forest Ensemble"
        else:
            final_probs = probs_svm
            active_model_name = best_name

        top_class_idx = int(np.argmax(final_probs))
        predicted_genre = classes[top_class_idx]
        prob_dict = {cls: round(float(p) * 100, 2) for cls, p in zip(classes, final_probs)}
        sorted_probs = dict(sorted(prob_dict.items(), key=lambda item: item[1], reverse=True))
    else:
        predicted_genre = model.predict(X_scaled_regularized)[0]
        active_model_name = best_name
        sorted_probs = {}

    result = {
        "audio_file": audio_file.name,
        "predicted_genre": predicted_genre,
        "model_used": active_model_name,
        "probabilities": sorted_probs,
        "metrics": {
            "tempo": round(float(raw_features.get("tempo", 0)), 1),
            "brightness_hz": round(float(raw_features.get("spectral_centroid_mean", 0)), 1),
            "energy_rms": round(float(raw_features.get("rms_mean", 0)), 4),
            "zcr": round(float(raw_features.get("zcr_mean", 0)), 4)
        }
    }

    return result


def main():
    parser = argparse.ArgumentParser(description="Predict Music Genre from Audio Track")
    parser.add_argument("audio_path", type=str, help="Path to audio file (.mp3 or .wav)")
    parser.add_argument("--models_dir", type=str, default="models", help="Directory where model artifacts are stored")
    args = parser.parse_args()

    try:
        prediction = predict_genre(args.audio_path, args.models_dir)
        print("\n" + "=" * 50)
        print(f" PREDICTION RESULT: {prediction['predicted_genre'].upper()}")
        print("=" * 50)
        
        if prediction["probabilities"]:
            print("\nGenre Confidence Probabilities:")
            for genre, prob in prediction["probabilities"].items():
                bar = "#" * int(prob / 5)
                print(f"  {genre:<16} : {prob:>5.1f}%  {bar}")
        print("=" * 50 + "\n")
        
    except Exception as e:
        print(f"\n[Error] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
