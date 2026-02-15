from __future__ import annotations

from pathlib import Path
import os
import numpy as np
from basic_pitch import constants

BASE_DIR = Path(__file__).resolve().parents[1]
RUNTIME_DIR = BASE_DIR / "runtime"
CLIENTS_DIR = RUNTIME_DIR / "clients"
UPLOADS_DIRNAME = "uploads"
ARTIFACTS_DIRNAME = "artifacts"

MAX_CONTENT_LENGTH = int(os.getenv("SOLFEGE_MAX_UPLOAD_BYTES", str(200 * 1024 * 1024)))
ALLOWED_EXTENSIONS = {"mp3", "wav", "flac", "ogg", "m4a", "mp4"}
MAX_WORKERS = int(os.getenv("SOLFEGE_MAX_WORKERS", "1"))

MIDI_OFFSET = 21
HOP_LENGTH = constants.FFT_HOP
SAMPLE_RATE = constants.AUDIO_SAMPLE_RATE
SPF_THEORETICAL = HOP_LENGTH / SAMPLE_RATE
SUBDIVISIONS = 4

N_OCTAVES = 7

# Krumhansl-Schmuckler Key Profiles
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
KEY_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
FULL_KEY_LABELS = [f"{k} Major" for k in KEY_NAMES] + [f"{k} Minor" for k in KEY_NAMES]

# Key Estimation Transition Probabilities
KEY_ESTIMATION_HMM_PARAMS = {
    "STABLE_PROB": 0.90,
    "RELATIVE_PROB": 0.05,
    "FIFTH_PROB": 0.01,
    "PARALLEL_PROB": 0.003,
}

# 歌唱音域 (basic-pitch インデックス)
LOWEST_PITCH = 15   # C2 (MIDI 36)
HIGHEST_PITCH = 67  # E6 (MIDI 88)

# HMM パラメータ
PITCH_STAY_PROB = 0.05
PITCH_TO_REST_PROB = 0.10
REST_STAY_PROB = 0.05
JUMP_SIGMA = 8.0
KEY_IN_WEIGHT = 1.0
KEY_OUT_WEIGHT = 0.8

# 再帰行列パラメータ
REC_WIDTH = 3
REC_MIN_LEN = 8
REC_QUANTILE = 0.90