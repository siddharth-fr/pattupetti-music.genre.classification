"""
pattupetti — Web Application for Music Genre Classification

Features:
- Minimal aesthetic waveform player with interactive scrubber needle
- Audio file upload endpoint (/api/predict)
- Live microphone recording support with WebM/Opus decoding (/api/predict)
- Pre-bundled sample tracks for instant testing without 8GB download (/api/samples)
- Audio streaming endpoint (/api/sample-audio/<track_id>)
- Comprehensive project documentation report (/docs)
"""

import os
import uuid
import time
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
from src.predict import predict_genre
from src.dataset import get_audio_path, AUDIO_DIR

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
CORS(app)

TEMP_DIR = Path("data/temp_audio")
TEMP_DIR.mkdir(parents=True, exist_ok=True)
SAMPLES_DIR = Path("data/samples")
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

# Curated sample tracks for viva demo
SAMPLE_TRACKS = [
    {"id": 2, "genre": "Hip-Hop", "title": "Awaken", "artist": "Sample 01"},
    {"id": 140, "genre": "Folk", "title": "Acoustic Melody", "artist": "Sample 02"},
    {"id": 182, "genre": "Rock", "title": "Electric Drive", "artist": "Sample 03"},
    {"id": 1482, "genre": "Electronic", "title": "Synthetic Waves", "artist": "Sample 04"},
]


@app.route("/")
def index():
    """Serves the main application page."""
    return render_template("index.html")


@app.route("/docs")
def docs():
    """Serves the project documentation report page."""
    return render_template("docs.html")


@app.route("/api/samples", methods=["GET"])
def get_samples():
    """Returns list of curated test tracks."""
    return jsonify({"status": "success", "samples": SAMPLE_TRACKS})


@app.route("/api/sample-audio/<int:track_id>", methods=["GET"])
def stream_sample_audio(track_id):
    """Streams a sample audio track for playback."""
    # Check bundled samples first
    bundled_path = SAMPLES_DIR / f"{track_id:06d}.mp3"
    if bundled_path.exists():
        return send_file(bundled_path, mimetype="audio/mpeg")

    # Fallback to raw dataset if present
    audio_path = get_audio_path(AUDIO_DIR, track_id)
    if not audio_path.exists():
        alt_path = get_audio_path(AUDIO_DIR / "fma_small", track_id)
        if alt_path.exists():
            audio_path = alt_path
        else:
            return jsonify({"status": "error", "message": "Audio file not found"}), 404
            
    return send_file(audio_path, mimetype="audio/mpeg")


@app.route("/api/predict", methods=["POST"])
def predict():
    """
    Accepts an uploaded file or audio blob from mic recording,
    runs the trained ML model, and returns predicted genre and probabilities.
    """
    if "audio" not in request.files:
        return jsonify({"status": "error", "message": "No audio file provided."}), 400

    audio_file = request.files["audio"]
    if audio_file.filename == "":
        return jsonify({"status": "error", "message": "Empty audio file received."}), 400

    # Generate unique temp filename preserving extension
    ext = os.path.splitext(audio_file.filename)[1].lower()
    if not ext:
        ext = ".wav"  # Default for browser recorded audio blobs

    temp_filename = f"temp_{int(time.time())}_{uuid.uuid4().hex[:8]}{ext}"
    temp_path = TEMP_DIR / temp_filename

    try:
        audio_file.save(str(temp_path))
        file_size = temp_path.stat().st_size
        print(f"\n[API /predict] Received audio: '{audio_file.filename}' ({file_size} bytes)", flush=True)

        # Run model inference
        result = predict_genre(str(temp_path))
        top_prob = list(result['probabilities'].items())[0] if result.get('probabilities') else ('N/A', 0)
        print(f"[API /predict] Output -> {result['predicted_genre']} ({top_prob[0]}: {top_prob[1]}%)", flush=True)

        return jsonify({
            "status": "success",
            "predicted_genre": result["predicted_genre"],
            "model_used": result["model_used"],
            "probabilities": result["probabilities"],
            "metrics": result.get("metrics", {})
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[API /predict ERROR] {type(e).__name__}: {e}", flush=True)
        return jsonify({"status": "error", "message": f"{type(e).__name__}: {str(e)}"}), 500

    finally:
        # Clean up temporary file
        if temp_path.exists():
            try:
                temp_path.unlink()
            except Exception:
                pass


@app.route("/api/predict-sample/<int:track_id>", methods=["GET"])
def predict_sample(track_id):
    """Predicts directly from a known sample track ID."""
    bundled_path = SAMPLES_DIR / f"{track_id:06d}.mp3"
    if bundled_path.exists():
        audio_path = bundled_path
    else:
        audio_path = get_audio_path(AUDIO_DIR, track_id)
        if not audio_path.exists():
            alt_path = get_audio_path(AUDIO_DIR / "fma_small", track_id)
            if alt_path.exists():
                audio_path = alt_path
            else:
                return jsonify({"status": "error", "message": "Sample file not found"}), 404

    try:
        result = predict_genre(str(audio_path))
        return jsonify({
            "status": "success",
            "predicted_genre": result["predicted_genre"],
            "model_used": result["model_used"],
            "probabilities": result["probabilities"],
            "metrics": result.get("metrics", {})
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print(" MUSIC GENRE CLASSIFICATION - WEB APP RUNNING")
    print(" Access URL: http://127.0.0.1:5000")
    print("=" * 60 + "\n")
    app.run(host="127.0.0.1", port=5000, debug=False)
