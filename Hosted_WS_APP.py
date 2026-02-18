from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import whisper
import tempfile
import os
import traceback
import uvicorn

app = FastAPI(title="Whisper Transcription API")

print("Loading Whisper model...")
model = whisper.load_model("base")
print("Model ready.")

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    tmp_path = None

    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file uploaded")

        # Save uploaded file safely
        suffix = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name
            contents = await file.read()
            tmp.write(contents)
            tmp.flush()
            os.fsync(tmp.fileno())

        if os.path.getsize(tmp_path) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")

        # Direct transcription (no manual WAV conversion)
        result = model.transcribe(tmp_path)

        return JSONResponse({
            "filename": file.filename,
            "language": result.get("language"),
            "duration_estimate": len(result.get("segments", [])),
            "text": result.get("text"),
            "segments": result.get("segments")
        })

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": "Internal server error", "details": str(e)})

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
