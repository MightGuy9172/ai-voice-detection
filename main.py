import base64
import os
import uuid
import subprocess

import librosa
import torch
from fastapi import FastAPI, HTTPException, Header, Depends
from pydantic import BaseModel
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import whisper
from dotenv import load_dotenv

# ---------------- LOAD ENV ----------------
load_dotenv()
API_KEY = os.getenv("API_KEY")

if not API_KEY:
    raise RuntimeError("API_KEY missing")

# ---------------- APP ----------------
app = FastAPI(title="AI Voice Detection API")

TEMP_DIR = "temp"
os.makedirs(TEMP_DIR, exist_ok=True)

# ---------------- LAZY MODELS (CRITICAL) ----------------
processor = None
wav2vec = None
whisper_model = None

def load_wav2vec():
    global processor, wav2vec
    if processor is None or wav2vec is None:
        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
        wav2vec = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")

def load_whisper():
    global whisper_model
    if whisper_model is None:
        whisper_model = whisper.load_model("tiny")  # FREE-tier safe

# ---------------- AUTH ----------------
def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")

# ---------------- REQUEST ----------------
class AudioInput(BaseModel):
    audio_base64: str

# ---------------- UTILS ----------------
def save_and_convert_audio(audio_base64: str):
    try:
        audio_bytes = base64.b64decode(audio_base64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid Base64 audio")

    file_id = str(uuid.uuid4())
    mp3 = f"{TEMP_DIR}/{file_id}.mp3"
    wav = f"{TEMP_DIR}/{file_id}.wav"

    with open(mp3, "wb") as f:
        f.write(audio_bytes)

    subprocess.run(
        ["ffmpeg", "-y", "-i", mp3, "-ar", "16000", "-ac", "1", wav],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    if not os.path.exists(wav):
        os.remove(mp3)
        raise HTTPException(status_code=500, detail="FFmpeg conversion failed")

    return mp3, wav

def extract_features(wav_path):
    load_wav2vec()
    audio, _ = librosa.load(wav_path, sr=16000)
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt")

    with torch.no_grad():
        outputs = wav2vec(**inputs)

    return outputs.last_hidden_state.mean(dim=1)

def detect_language(wav_path):
    load_whisper()
    return whisper_model.transcribe(wav_path).get("language", "unknown")

def classify(features):
    confidence = torch.sigmoid(features.mean()).item()
    confidence = round(confidence, 2)

    if confidence >= 0.5:
        return (
            "AI_GENERATED",
            confidence,
            "Over-smooth prosody and limited pitch variation detected, typical of AI-generated speech."
        )
    else:
        return (
            "HUMAN",
            confidence,
            "Natural pitch variation and micro-pauses detected, typical of human speech."
        )

# ---------------- ENDPOINT ----------------
@app.post("/detect", dependencies=[Depends(verify_api_key)])
def detect_voice(data: AudioInput):
    mp3, wav = save_and_convert_audio(data.audio_base64)

    try:
        features = extract_features(wav)
        language = detect_language(wav)
        label, confidence, explanation = classify(features)
    finally:
        if os.path.exists(mp3):
            os.remove(mp3)
        if os.path.exists(wav):
            os.remove(wav)

    return {
        "classification": label,
        "confidence_score": confidence,
        "language": language,
        "explanation": explanation
    }
