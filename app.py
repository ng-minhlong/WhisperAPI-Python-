from fastapi import FastAPI, UploadFile, File
from faster_whisper import WhisperModel
import tempfile
import os

app = FastAPI()

# dùng tiny để tránh crash free tier
MODEL_SIZE = os.getenv("MODEL_SIZE", "tiny")

print(f"Loading model: {MODEL_SIZE}")
model = WhisperModel(MODEL_SIZE, compute_type="int8")

@app.get("/")
def root():
    return {"status": "ok"}

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    segments, _ = model.transcribe(tmp_path)

    text = " ".join([seg.text for seg in segments])
    return {"text": text}