from __future__ import annotations

from dataclasses import dataclass

NOTE_NAMES_SHARP = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
MAJOR_PROFILE = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
MINOR_PROFILE = [6.33, 2.68, 3.52, 5.38, 2.6, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]

MOVABLE_DO_JA_SHARP = ["ド", "ド#", "レ", "レ#", "ミ", "ファ", "ファ#", "ソ", "ソ#", "ラ", "ラ#", "シ"]


def pitch_to_movable_do(pitch: int, key_root_pc: int) -> str:
    interval = (pitch % 12 - key_root_pc) % 12
    return MOVABLE_DO_JA_SHARP[interval]


def attach_solfege(note_events: list[dict], key) -> list[dict]:
    """ノートイベントにソルフェージュラベルを付与する。

    key は root_pc 属性を持つオブジェクト
    (key_estimation.KeyEstimate または旧 solfege.KeyEstimate いずれも可)。
    """
    out = []
    for ev in note_events:
        copied = dict(ev)
        copied["solfege"] = pitch_to_movable_do(int(ev["pitch"]), key.root_pc)
        out.append(copied)
    return out
