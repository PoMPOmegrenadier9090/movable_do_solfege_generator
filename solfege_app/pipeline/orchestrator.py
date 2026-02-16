from __future__ import annotations

from pathlib import Path
from typing import Callable

from .separation import AudioSeparationPipeline, DemucsSeparationResult
from .transcription import run_transcription


ProgressFn = Callable[..., None]


def run_full_pipeline(
    job_id: str,
    update_job: ProgressFn,
    client_id: str,
    job_dir: Path,
    input_path: Path,
) -> dict:
    artifacts_dir = job_dir / "artifacts"
    separation = AudioSeparationPipeline(input_audio=input_path, job_dir=job_dir)

    # Demucsでボーカルを抽出する
    update_job(job_id, step="demucs", progress=0.1, message="Running Demucs separation")
    demucs_result: DemucsSeparationResult = separation.run_demucs()

    update_job(job_id, step="transcription", progress=0.75, message="Running transcription + solfege")
    transcription = run_transcription(
        demucs_result, artifacts_dir,
        original_audio_path=input_path,
    )

    update_job(job_id, step="finalize", progress=0.95, message="Preparing result")
    return {
        "job_id": job_id,
        "client_id": client_id,
        "estimated_global_key": transcription["estimated_global_key"],
        "note_count": transcription["note_count"],
        "media": {
            "midi": f"/api/media/{client_id}/{job_id}/{transcription['midi_filename']}",
            "mix_audio": f"/api/media/{client_id}/{job_id}/{transcription['mix_filename']}",
            "solfege_json": f"/api/media/{client_id}/{job_id}/{transcription['solfege_filename']}",
        },
        "notes": transcription["notes"],
        "key_sequence": transcription["key_sequence"],
    }
