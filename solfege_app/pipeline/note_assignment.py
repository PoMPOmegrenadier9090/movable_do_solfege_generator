"""動的 Viterbi HMM によるノート推定モジュール。

キー系列に依存した遷移行列を用いて最適なノート系列を推定する。
"""
from __future__ import annotations

import numpy as np
import pretty_midi

from .. import config

def greedy_note_assignment(
    grid_onsets: np.ndarray,
    grid_notes: np.ndarray,
    grid_times: np.ndarray,
    onset_threshold: float = 0.50,
    pitch_threshold: float = 0.45,
) -> tuple[np.ndarray, list[pretty_midi.Note]]:
    """グリーディなノート割り当てを行い、割り当て済みピッチ配列と MIDI ノートを返す。"""
    if grid_onsets.shape != grid_notes.shape:
        raise ValueError("grid_onsets and grid_notes must have the same shape")
    if grid_onsets.shape[0] != len(grid_times):
        raise ValueError("grid count and grid_times length must match")

    num_grids = grid_onsets.shape[0]
    assigned = grid_notes.copy()
    notes: list[pretty_midi.Note] = []

    # 走査中のグリッドで鳴っている音
    active_pitch: int | None = None
    # 鳴っている音の開始時刻
    active_start: float | None = None

    for i in range(num_grids):
        onset = grid_onsets[i]
        pitch = grid_notes[i]
        grid_time = float(grid_times[i])

        detected_onset = (int(np.argmax(onset)) + config.MIDI_OFFSET) if np.max(onset) >= onset_threshold else None
        detected_pitch = (int(np.argmax(pitch)) + config.MIDI_OFFSET) if np.max(pitch) >= pitch_threshold else None

        # 終了判定
        if active_pitch is not None: # すでに音が鳴っている場合
            should_close = (
                detected_pitch is None
                or detected_pitch != active_pitch
                or detected_onset is not None
            )
            if should_close:
                notes.append(pretty_midi.Note(
                    velocity=50, pitch=int(active_pitch),
                    start=active_start, end=grid_time,
                ))
                active_pitch = None
                active_start = None

        # 開始判定
        if active_pitch is None and detected_pitch is not None:
            active_pitch = detected_pitch
            active_start = grid_time

        # 現在のグリッドで音が鳴っている場合，そのピッチを割り当て
        if active_pitch is not None:
            assigned[i] = np.zeros_like(assigned[i])
            assigned[i][active_pitch - config.MIDI_OFFSET] = 1.0

    # 末尾処理
    if active_pitch is not None:
        end_time = grid_times[-1] + (grid_times[-1] - grid_times[-2]) if num_grids > 1 else grid_times[-1]
        notes.append(pretty_midi.Note(
            velocity=50,
            pitch=int(active_pitch),
            start=active_start,
            end=float(end_time)
        ))

    return assigned, notes
    

def note_list_to_midi(midi_notes: list[pretty_midi.Note]) -> pretty_midi.PrettyMIDI:
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)
    instrument.notes = midi_notes
    pm.instruments.append(instrument)
    return pm