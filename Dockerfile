FROM python:3.10-slim

WORKDIR /app

# system deps
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# preload model (để tránh runtime download)
RUN python -c "from faster_whisper import WhisperModel; print('loading model...'); WhisperModel('base', compute_type='int8')"
COPY . .

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]