import base64
import os
import uuid
import numpy as np
import librosa

from fastapi import FastAPI, HTTPException, Header, Depends
from pydantic import BaseModel
from dotenv import load_dotenv

# ---------------- LOAD ENV ----------------
load_dotenv()
API_KEY = os.getenv("API_KEY")

if not API_KEY:
    raise RuntimeError("API_KEY missing")

# ---------------- APP ----------------
app = FastAPI(title="AI Voice Detection API (Lightweight)")

TEMP_DIR = "temp"
os.makedirs(TEMP_DIR, exist_ok=True)

# ---------------- AUTH ----------------
def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")

# ---------------- REQUEST MODEL ----------------
class AudioInput(BaseModel):
    language: str | None = None
    audioFormat: str | None = None
    audioBase64: str

# ---------------- UTILS ----------------
def save_audio(audio_base64: str):
    try:
        audio_bytes = base64.b64decode(audio_base64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid Base64 audio")

    file_path = os.path.join(TEMP_DIR, f"{uuid.uuid4()}.mp3")
    with open(file_path, "wb") as f:
        f.write(audio_bytes)

    return file_path

def extract_features(audio_path):
    y, sr = librosa.load(audio_path, sr=16000)

    # Pitch variation
    pitches, _ = librosa.piptrack(y=y, sr=sr)
    pitch_values = pitches[pitches > 0]
    pitch_variance = np.var(pitch_values) if len(pitch_values) > 0 else 0

    # Energy variation
    energy_variance = np.var(librosa.feature.rms(y=y))

    # Spectral flatness
    flatness = np.mean(librosa.feature.spectral_flatness(y=y))

    # Silence ratio
    intervals = librosa.effects.split(y, top_db=25)
    silence_ratio = 1 - (np.sum(intervals[:, 1] - intervals[:, 0]) / len(y))

    return pitch_variance, energy_variance, flatness, silence_ratio

def classify(features):
    pitch_var, energy_var, flatness, silence_ratio = features

    score = (
        0.4 * (1 / (1 + pitch_var)) +
        0.3 * (1 / (1 + energy_var)) +
        0.2 * flatness +
        0.1 * silence_ratio
    )

    confidence = round(float(min(max(score, 0.0), 1.0)), 2)

    if confidence >= 0.5:
        return (
            "AI_GENERATED",
            confidence,
            "Low pitch variation and smooth energy patterns detected, which are commonly observed in AI-generated speech."
        )
    else:
        return (
            "HUMAN",
            confidence,
            "Natural pitch variation and irregular energy patterns detected, typical of human speech."
        )

# ---------------- ENDPOINT ----------------
@app.post("/detect", dependencies=[Depends(verify_api_key)])
def detect_voice(data: AudioInput):
    audio_path = save_audio(data.audioBase64)

    try:
        features = extract_features(audio_path)
        label, confidence, explanation = classify(features)
    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)

    return {
        "classification": label,
        "confidence_score": confidence,
        "language": data.language or "unknown",
        "explanation": explanation
    }
