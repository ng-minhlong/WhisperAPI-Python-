import asyncio
import os
import shutil
import tempfile

import imageio_ffmpeg
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel

# Make ffmpeg available without system install
ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
ffmpeg_dir = os.path.dirname(ffmpeg_exe)
os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_SIZE = os.getenv("MODEL_SIZE", "tiny")   # tiny để free tier đỡ chết
LANGUAGE = os.getenv("LANGUAGE", "vi")         # đổi sang "" nếu muốn auto-detect
MAX_CONCURRENT = int(os.getenv("MAX_CONCURRENT", "1"))

model = WhisperModel(MODEL_SIZE, compute_type="int8")
semaphore = asyncio.Semaphore(MAX_CONCURRENT)


@app.get("/")
def root():
    return {
        "status": "ok",
        "model": MODEL_SIZE,
        "language": LANGUAGE,
    }


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    tmp_path = None

    try:
        suffix = os.path.splitext(file.filename or "")[1] or ".webm"

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
          shutil.copyfileobj(file.file, tmp)
          tmp_path = tmp.name

        async with semaphore:
            segments, info = model.transcribe(
                tmp_path,
                language=LANGUAGE if LANGUAGE else None,
                vad_filter=True,
                beam_size=1,
                condition_on_previous_text=False,
                temperature=0.0,
            )

            text = " ".join(seg.text.strip() for seg in segments).strip()

        return {
            "text": text,
            "language": info.language,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        try:
            await file.close()
        except Exception:
            pass

        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass