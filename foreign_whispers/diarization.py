"""Speaker diarization using pyannote.audio.

Extracted from notebooks/foreign_whispers_pipeline.ipynb (M2-align).

Optional dependency: pyannote.audio
    pip install pyannote.audio
Requires accepting the pyannote/speaker-diarization-3.1 licence on HuggingFace
and providing an HF token.  Returns empty list with a warning if the dep is
absent or the token is missing.
"""
import logging

logger = logging.getLogger(__name__)


def diarize_audio(audio_path: str, hf_token: str | None = None) -> list[dict]:
    """Return speaker-labeled intervals for *audio_path*.

    Returns:
        List of ``{start_s: float, end_s: float, speaker: str}``.
        Empty list when pyannote.audio is absent, token is missing, or diarization fails.
    """
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
        diarization = pipeline(audio_path)
        
        results = []
        
        # SAFETY CHECK: If it's the expected full Annotation object
        if hasattr(diarization, "itertracks"):
            for speech_turn, _, speaker in diarization.itertracks(yield_label=True):
                results.append({
                    "start_s": float(speech_turn.start),
                    "end_s": float(speech_turn.end),
                    "speaker": str(speaker)
                })
        # FALLBACK: If it's a simplified dictionary/DiarizeOutput
        elif hasattr(diarization, "items"):
            for speech_turn, speaker in diarization.items():
                results.append({
                    "start_s": float(speech_turn.start),
                    "end_s": float(speech_turn.end),
                    "speaker": str(speaker)
                })
        
        return results

    except Exception as exc:
        logger.error(f"Diarization failed completely: {exc}")
        return []

def diarize(segments: list[dict], diarization_output: list[dict]) -> list[dict]:
    for part in segments:
        best_speaker=None
        max_overlap=0.0
        for segment in diarization_output:
            speaker_start=segment['start_s']
            speaker_end=segment['end_s']
            seg_start=part['start']
            seg_end=part['end']
            current_overlap=max(0,min(seg_end,speaker_end) - max(seg_start,speaker_start))
            if current_overlap > max_overlap:
                max_overlap = current_overlap
                best_speaker=segment['speaker']
        part['speaker'] = best_speaker

    return segments
     