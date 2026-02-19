from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import whisper
import tempfile
import os
import traceback
import subprocess
import base64
import re

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

        # ---- 1) Clean + Decode Base64 (Salesforce safe) ----
        base64_data = payload.audio_base64.split(",")[-1]

        # Remove whitespace/newlines that Salesforce may insert
        base64_data = re.sub(r"\s+", "", base64_data)

        # Fix missing padding
        missing_padding = len(base64_data) % 4
        if missing_padding:
            base64_data += "=" * (4 - missing_padding)

        try:
            audio_bytes = base64.b64decode(base64_data)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid base64 audio")

        if len(audio_bytes) < 1000:
            raise HTTPException(status_code=400, detail="Decoded audio too small")

        # ---- 2) Save original file ----
        suffix = os.path.splitext(payload.filename)[1] or ".webm"

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
            input_path = f.name
            f.write(audio_bytes)

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

        # ---- 4) Transcribe ----
        result = model.transcribe(converted_path)

        return JSONResponse({
            "filename": payload.filename,
            "language": result.get("language"),
            "text": result.get("text"),
            "segments": result.get("segments"),
            "duration_estimate": len(result.get("segments", []))
        })

    except HTTPException:
        raise

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