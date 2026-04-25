"""POST /api/diarize/{video_id} — speaker diarization (issue fw-lua)."""

import asyncio
import json
import subprocess

from fastapi import APIRouter, HTTPException

from api.src.core.config import settings
from api.src.core.dependencies import resolve_title
from api.src.schemas.diarize import DiarizeResponse
from api.src.services.alignment_service import AlignmentService

router = APIRouter(prefix="/api")

_alignment_service = AlignmentService(settings=settings)


@router.post("/diarize/{video_id}", response_model=DiarizeResponse)
async def diarize_endpoint(video_id: str):
    """Run speaker diarization on a video's audio track.

    Steps:
    1. Extract audio from video via ffmpeg
    2. Run pyannote diarization
    3. Cache and return speaker segments
    """
    title = resolve_title(video_id)
    if title is None:
        raise HTTPException(status_code=404, detail=f"Video {video_id} not found")

    diar_dir = settings.diarizations_dir
    diar_dir.mkdir(parents=True, exist_ok=True)
    diar_path = diar_dir / f"{title}.json"

    # Return cached result
    if diar_path.exists():
        data = json.loads(diar_path.read_text())
        return DiarizeResponse(
            video_id=video_id,
            speakers=data.get("speakers", []),
            segments=data.get("segments", []),
            skipped=True,
        )

    # ---- YOUR CODE HERE ----
    # Step 1: Extract audio from video
    video_path = settings.videos_dir / f"{title}.mp4"
    audio_path = diar_dir / f"{title}.wav"
    #   Use subprocess.run to call:
    #     ffmpeg -i <video_path> -vn -acodec pcm_s16le -ar 16000 -y <audio_path>
    command = [
        'ffmpeg', 
        '-i', str(video_path), 
        '-vn', 
        '-acodec', 'pcm_s16le', 
        '-ar', '16000',
        '-y',
        str(audio_path)
    ]

    try:
        result_ffmpeg = subprocess.run(command, check=True, capture_output=True, text=True)
        print("Command executed successfully!")
    except subprocess.CalledProcessError as e:
       raise HTTPException(status_code=500, detail=f"Error in audio extraction: {e.stderr}")
    
    # Step 2: Run diarization
    diar_segments = _alignment_service.diarize(str(audio_path))
    #
    # Step 3: Extract unique speakers
    speakers = sorted(set(s["speaker"] for s in diar_segments))
    #
    # Step 4: Cache result
  
    result = {"speakers": speakers, "segments": diar_segments}
    diar_path.write_text(json.dumps(result))
    #

    transcription_path = settings.transcriptions_dir / f"{title}.json"
    if transcription_path.exists():
            transcripted_data = json.loads(transcription_path.read_text())
            updated_segments = _alignment_service.assign_speakers(
                transcripted_data["segments"],
                diar_segments
            )
            transcripted_data["segments"] = updated_segments
            diar_trans_path = settings.transcriptions_dir / f"{title}_diarized.json"
            diar_trans_path.write_text(json.dumps(transcripted_data))

    # Step 5: Return DiarizeResponse
    return DiarizeResponse(video_id=video_id, speakers=speakers, segments=diar_segments)
    #
    #raise HTTPException(status_code=501, detail="Diarization not yet implemented")
    # ---- END YOUR CODE ----
