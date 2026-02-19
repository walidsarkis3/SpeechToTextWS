from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import whisper
import tempfile
import os
import traceback
import uvicorn


app = FastAPI(title="Whisper Transcription API")

print("Loading Whisper model...")
model = whisper.load_model("base")
print("Model ready.")

class Base64Audio(BaseModel):
    filename: str
    audio_base64: str


@app.post("/transcribe")
async def transcribe_audio(payload: Base64Audio):
    input_path = None
    converted_path = None

    try:
        if not payload.audio_base64:
            raise HTTPException(status_code=400, detail="No audio data")

        # ---- 1) Decode Base64 ----
        try:
            audio_bytes = base64.b64decode(payload.audio_base64)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid base64 audio")

        # ---- 2) Save original file ----
        suffix = os.path.splitext(payload.filename)[1] or ".webm"

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
            input_path = f.name
            f.write(audio_bytes)
            f.flush()
            os.fsync(f.fileno())

        if os.path.getsize(input_path) == 0:
            raise HTTPException(status_code=400, detail="Decoded file is empty")

        # ---- 3) Convert to MP3 using ffmpeg ----
        converted_path = input_path + ".mp3"

        cmd = [
            "ffmpeg",
            "-y",
            "-i", input_path,
            "-vn",
            "-acodec", "libmp3lame",
            "-ar", "16000",
            "-ac", "1",
            converted_path
        ]

        process = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        if process.returncode != 0:
            print(process.stderr.decode())
            raise HTTPException(status_code=500, detail="Audio conversion failed (ffmpeg)")

        # ---- 4) Transcribe with Whisper ----
        result = model.transcribe(converted_path)

        return JSONResponse({
            "filename": payload.filename,
            "language": result.get("language"),
            "text": result.get("text"),
            "segments": result.get("segments"),
            "duration_estimate": len(result.get("segments", []))
        })

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error", "details": str(e)}
        )

    finally:
        for path in [input_path, converted_path]:
            if path and os.path.exists(path):
                os.remove(path)
