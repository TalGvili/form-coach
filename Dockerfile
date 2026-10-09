# syntax=docker/dockerfile:1
# Form Coach, the web app, with everything it needs: Python 3.12, ffmpeg and the pose model.
#
#   docker build -t form-coach .
#   docker run -p 127.0.0.1:8000:8000 -v form-coach-data:/app/data form-coach
#
# The volume keeps data/ (the history database, annotated videos, landmark caches) across
# restarts and rebuilds. 127.0.0.1 keeps the server on this computer; drop it to let a phone on
# the same Wi-Fi connect, on a network you trust.

FROM python:3.12-slim

# ffmpeg encodes the annotated video for browsers. OpenCV loads libGL and GLib when imported;
# MediaPipe's native library loads EGL and GLES when the pose model is created. Only an upload
# does that (the tests never create the model), so a real upload in the container is the test
# for these two.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg libgl1 libglib2.0-0 libegl1 libgles2 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

# Dependencies before the code: Docker reuses this layer until requirements.txt changes, so
# editing code doesn't reinstall every package.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# The pose model (kept out of git), checked against its SHA-256 so a changed or corrupted
# download fails the build. A downloaded file is readable only by root unless told otherwise;
# --chmod would also apply to a folder ADD creates, and a folder without the execute bit can't
# be entered, so models/ is made first.
RUN mkdir models
ADD --checksum=sha256:5134a3aad27a58b93da0088d431f366da362b44e3ccfbe3462b3827a839011b1 \
    --chmod=644 \
    https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task \
    models/pose_landmarker_full.task

COPY . .

# Run as an ordinary user who can write only data/: if a malformed video ever exploited a bug in
# a video decoder, it would be stuck in a container with no admin rights.
RUN useradd --create-home coach && mkdir -p data && chown coach data
USER coach

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
