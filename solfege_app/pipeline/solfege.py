from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .. import config

NOTE_NAMES_SHARP = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
MAJOR_PROFILE = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
MINOR_PROFILE = [6.33, 2.68, 3.52, 5.38, 2.6, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]

MOVABLE_DO_JA_SHARP = ["ド", "ド#", "レ", "レ#", "ミ", "ファ", "ファ#", "ソ", "ソ#", "ラ", "ラ#", "シ"]


def key_index_to_root_pc(key_index: int) -> int:
    """キーインデックス (0–23) から root pitch class (0–11) を返す。"""
    return key_index % 12


def key_index_is_minor(key_index: int) -> bool:
    """キーインデックスがマイナーキーかどうかを返す。"""
    return key_index >= 12


def key_index_to_solfege_root(key_index: int) -> int:
    """ソルフェージュ用のルート pitch class を返す。

    メジャーキー: そのまま root_pc を返す（root = ド）。
    マイナーキー: 平行長調の root_pc を返す（短3度上 = ド）。
    例: A minor (root=9) → C major (root=0)
        F minor (root=5) → Ab major (root=8)
    """
    root_pc = key_index % 12
    if key_index >= 12:  # minor
        return (root_pc + 3) % 12
    return root_pc


def key_index_to_label(key_index: int) -> str:
    """キーインデックス (0–23) からキーラベル文字列を返す。"""
    return config.FULL_KEY_LABELS[key_index]


def pitch_to_movable_do(pitch: int, key_root_pc: int) -> str:
    interval = (pitch % 12 - key_root_pc) % 12
    return MOVABLE_DO_JA_SHARP[interval]


def _find_grid_index(note_start: float, grid_times: np.ndarray) -> int:
    """ノートの開始時間に対応するグリッドインデックスを返す。

    grid_times[i] <= note_start < grid_times[i+1] となる i を返す。
    最後のグリッド以降のノートは最後のグリッドに対応させる。
    """
    idx = int(np.searchsorted(grid_times, note_start, side="right")) - 1
    return max(0, min(idx, len(grid_times) - 1))


def attach_solfege(
    note_events: list[dict],
    key_sequence: np.ndarray,
    grid_times: np.ndarray,
) -> list[dict]:
    """ノートイベントにソルフェージュラベルを付与する（グリッドごとのキーを使用）。

    Parameters
    ----------
    note_events : ノートイベントのリスト。各要素は start, end, pitch, velocity を含む。
    key_sequence : (num_grids,) 各グリッドのキーインデックス (0–23)。
    grid_times : (num_grids,) 各グリッドの開始時間（秒）。

    Returns
    -------
    ソルフェージュラベルとキー情報を追加したノートイベントのリスト。
    """
    out = []
    for ev in note_events:
        copied = dict(ev)
        grid_idx = _find_grid_index(ev["start"], grid_times)
        key_idx = int(key_sequence[grid_idx])
        root_pc = key_index_to_solfege_root(key_idx)
        copied["solfege"] = pitch_to_movable_do(int(ev["pitch"]), root_pc)
        copied["key"] = key_index_to_label(key_idx)
        out.append(copied)
    return out
