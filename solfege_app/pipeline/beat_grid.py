"""ビートグリッド量子化モジュール。

basic-pitch のフレーム単位の出力をビートグリッドに量子化する。
"""
from __future__ import annotations

import numpy as np
import librosa
from basic_pitch import constants as bp_constants
from .. import config

HOP_LENGTH = bp_constants.FFT_HOP
SAMPLING_RATE = bp_constants.AUDIO_SAMPLE_RATE
SECONDS_PER_FRAME = HOP_LENGTH / SAMPLING_RATE


def center_weighted_agg_func(array: np.ndarray, axis: int = 0) -> np.ndarray:
    """配列の中心に重みを置いて集約する (ハミング窓)。"""
    length = array.shape[axis]
    weights = np.hamming(length)
    if axis == 0:
        weighted = array * weights[:, np.newaxis]
    else:
        weighted = array * weights[np.newaxis, :]
    return np.sum(weighted, axis=axis) / np.sum(weights)

def detect_beats(
    y: np.ndarray, sr: int, hop_length: int = HOP_LENGTH
) -> tuple[float, np.ndarray]:
    """
    テンポとビートフレーム位置を検出する。
    Returns:
        midi_tempo: estimated tempo in BPM
        beat_frames: array of beat positions in frames
    """
    tempo_arr = librosa.feature.tempo(y=y, sr=sr)
    if tempo_arr.size:
        start_bpm = float(tempo_arr[0])
    else:
        print(f"Tempo cannot be detected. Using default tempo: 120.0 BPM")
        raise ValueError("Tempo cannot be detected.")
        # start_bpm = 120.0
    tempo, beat_frames = librosa.beat.beat_track(
        y=y, sr=sr, hop_length=hop_length, start_bpm=start_bpm, units="frames"
    )
    midi_tempo = float(tempo[0]) if np.asarray(tempo).size else start_bpm
    print(f"Detected tempo: {float(tempo[0])} BPM, {len(beat_frames)} beats")
    return midi_tempo, beat_frames


def quantize_to_grid(
    data: np.ndarray,
    beat_times_sec: np.ndarray,
    spf: float,
    subdivisions: int = 4,
    agg_func=np.mean,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    フレームデータをビートグリッドに量子化する（実時間ベース）．
    data: (num_frames, num_features)
    beat_times_sec: ビートの実時間（秒）配列 (float)
    spf: seconds per frame (データ配列のフレーム→秒変換用)
    subdivisions: 1ビートあたりの分割数
    agg_func: 集約関数 (np.max, np.mean等). signature: func(array, axis=int)
    Returns: (grid_values, grid_times) - grid_times は秒（float）
    """
    end_frame = data.shape[0]
    total_duration = end_frame * spf
    grid_values = []
    grid_times = []  # 実時間（秒）
    prev_beat_time = 0.0

    for beat_time in list(beat_times_sec):
        # ビート間をsubdivisionsで等分割（秒単位、floatのまま）
        sub_times = np.linspace(prev_beat_time, beat_time, subdivisions + 1)
        for j in range(subdivisions):
            t_start, t_end = sub_times[j], sub_times[j + 1]
            if t_end <= t_start:
                continue
            # 実時間からフレームインデックスに変換（集約範囲の特定のみ）
            f_start = int(round(t_start / spf))
            f_end = int(round(t_end / spf))
            f_start = min(f_start, end_frame)
            f_end = min(f_end, end_frame)

            if f_start >= end_frame or f_end <= f_start:
                grid_values.append(np.zeros(data.shape[1]))
            else:
                chunk = data[f_start:f_end]
                if chunk.shape[0] == 0:
                    grid_values.append(np.zeros(data.shape[1]))
                else:
                    grid_values.append(agg_func(chunk, axis=0))
            grid_times.append(t_start)  # グリッドの正確な実時間を保持
        prev_beat_time = beat_time

    # 最後のビート以降
    if prev_beat_time < total_duration:
        sub_times = np.linspace(prev_beat_time, total_duration, subdivisions + 1)
        for j in range(subdivisions):
            t_start, t_end = sub_times[j], sub_times[j + 1]
            if t_end <= t_start:
                continue
            f_start = int(round(t_start / spf))
            f_end = int(round(t_end / spf))
            f_start = min(f_start, end_frame)
            f_end = min(f_end, end_frame)
            if f_start < end_frame and f_end > f_start:
                grid_values.append(agg_func(data[f_start:f_end], axis=0))
            else:
                grid_values.append(np.zeros(data.shape[1]))
            grid_times.append(t_start)

    return np.array(grid_values), np.array(grid_times, dtype=np.float64)