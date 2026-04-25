"""Speaker diarization using pyannote.audio."""
import logging

logger = logging.getLogger(__name__)

def diarize_audio(audio_path: str, hf_token: str | None = None) -> list[dict]:
    """Return speaker-labeled intervals for *audio_path*."""
    # SECURITY: Token is now passed via arguments or environment, not hardcoded.
    if not hf_token:
        logger.warning("No HF token provided — diarization skipped.")
        return []

    try:
        from pyannote.audio import Pipeline
    except (ImportError, TypeError):
        logger.warning("pyannote.audio not installed — returning empty diarization.")
        return []
    
    try:
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            token=hf_token
        )
        
        print("--- AI is starting processing... ---")
        diarization = pipeline(audio_path)
        
        results = []

        # Handle the Pyannote 3.1 'DiarizeOutput' Object structure
        if hasattr(diarization, "speaker_diarization"):
            target = diarization.speaker_diarization
        elif hasattr(diarization, "annotation"):
            target = diarization.annotation
        else:
            target = diarization

        try:
            for segment, _, label in target.itertracks(yield_label=True):
                results.append({
                    "start_s": float(segment.start),
                    "end_s": float(segment.end),
                    "speaker": str(label)
                })
        except Exception as e:
            print(f"DEBUG: Extraction error: {e}")
            if hasattr(target, "items"):
                for segment, label in target.items():
                    results.append({
                        "start_s": float(segment.start),
                        "end_s": float(segment.end),
                        "speaker": str(label)
                    })

        print(f"DEBUG: Final results count: {len(results)}")
        return results

    except Exception as exc:
        logger.error(f"Diarization failed completely: {exc}")
        return []

def diarize(segments: list[dict], diarization_output: list[dict]) -> list[dict]:
    """Overlap matching logic for Whisper segments and Pyannote speakers."""
    for part in segments:
        best_speaker = None
        max_overlap = 0.0
        for segment in diarization_output:
            speaker_start = segment['start_s']
            speaker_end = segment['end_s']
            seg_start = part['start']
            seg_end = part['end']
            current_overlap = max(0, min(seg_end, speaker_end) - max(seg_start, speaker_start))
            if current_overlap > max_overlap:
                max_overlap = current_overlap
                best_speaker = segment['speaker']
        part['speaker'] = best_speaker

    return segments