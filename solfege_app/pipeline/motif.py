"""モチーフ抽出・補正モジュール。

再帰行列からモチーフ区間を検出し、ノート確率を補正する。
"""
from __future__ import annotations

import numpy as np
import librosa
from scipy.ndimage import gaussian_filter1d, uniform_filter1d

from .. import config


def _extract_diagonal_runs(
    R_aff: np.ndarray,
    R_binary: np.ndarray
) -> list[dict]:
    """二値再帰行列の対角線上の連続区間を抽出する。"""
    n = R_binary.shape[0]
    segments: list[dict] = []

    for offset in range(config.REC_WIDTH, n):
        diag_bin = np.diagonal(R_binary, offset=offset)
        if diag_bin.size < config.REC_MIN_LEN:
            continue

        padded = np.concatenate(([0], diag_bin, [0]))
        diffs = np.diff(padded)
        starts = np.where(diffs == 1)[0]
        ends = np.where(diffs == -1)[0]

        for start, end in zip(starts, ends):
            length = end - start
            if length >= config.REC_MIN_LEN:
                diag_aff = np.diagonal(R_aff, offset=offset)
                score = float(np.mean(diag_aff[start:end]))
                segments.append({
                    "start_a": int(start),
                    "start_b": int(start + offset),
                    "length": int(length),
                    "score": score,
                })

    segments.sort(key=lambda s: (s["score"] * s["length"], s["length"]), reverse=True)
    return segments


def _overlap_1d(a0: int, a1: int, b0: int, b1: int) -> bool:
    return not (a1 <= b0 or b1 <= a0)


def _select_non_overlapping(segments: list[dict], top_k: int = 100) -> list[dict]:
    """重複しないモチーフ区間を選択する。"""
    selected: list[dict] = []
    for seg in segments:
        a0, a1 = seg["start_a"], seg["start_a"] + seg["length"]
        b0, b1 = seg["start_b"], seg["start_b"] + seg["length"]

        # A と B 自体が重なる場合は除外
        if _overlap_1d(a0, a1, b0, b1):
            continue

        ok = True
        # すでにkeptに追加されたモチーフと重複する場合は除外
        for kept in selected:
            ka0 = kept["start_a"]
            ka1 = ka0 + kept["length"]
            kb0 = kept["start_b"]
            kb1 = kb0 + kept["length"]
            if _overlap_1d(a0, a1, ka0, ka1) and _overlap_1d(b0, b1, kb0, kb1):
                ok = False
                break
        if ok:
            selected.append(seg)
        # k個選択したら終了
        if len(selected) >= top_k:
            break
    return selected


def extract_motifs(biased_notes: np.ndarray) -> list[dict]:
    """ノート確率からモチーフ区間を抽出する。

    Parameters
    ----------
    biased_notes : (num_grids, num_pitches) — スケールバイアス適用済みノート確率

    Returns
    -------
    motif_segments : モチーフ区間のリスト
    """
    # 歌唱音域に限定
    pitch_lo, pitch_hi = config.LOWEST_PITCH, config.HIGHEST_PITCH
    notes_in_range = np.array(
        biased_notes[:, pitch_lo:pitch_hi + 1], dtype=np.float64, copy=True
    )
    notes_in_range[notes_in_range < 0] = 0.0

    T, P = notes_in_range.shape
    if T < 2:
        return []

    profiles = notes_in_range.copy()

    # 平滑化
    profiles = gaussian_filter1d(profiles, sigma=0.8, axis=1, mode="nearest") # ピッチ方向
    profiles = uniform_filter1d(profiles, size=2, axis=0, mode="nearest") # 時間方向

    # ノイズのフィルタリング
    peak_values = profiles.max(axis=1, keepdims=True)
    profiles[profiles < (peak_values * 0.6)] = 0.0

    # 平坦フレーム抑制
    peak_strength = profiles.max(axis=1)
    threshold = np.mean(peak_values) * 0.7
    profiles[peak_strength < threshold] = 0.0

    # L2 正規化
    norms = np.linalg.norm(profiles, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    features = profiles / norms

    # 再帰行列
    R = librosa.segment.recurrence_matrix(
        features.T, k=None, width=config.REC_WIDTH,
        metric="cosine", mode="affinity", sym=True, sparse=False,
    )
    # 対角成分を強調
    R_enhanced = librosa.segment.timelag_filter(gaussian_filter1d)(R, sigma=1.2, mode="mirror")
    np.fill_diagonal(R_enhanced, 0.0)

    # しきい値
    valid_vals = R_enhanced[R_enhanced > 0]
    if valid_vals.size == 0:
        return []
    # 上位 REC_QUANTILE % の値をしきい値とする
    rec_threshold = np.percentile(valid_vals, config.REC_QUANTILE * 100)
    R_bin = (R_enhanced >= rec_threshold).astype(np.uint8)

    candidates = _extract_diagonal_runs(R_enhanced, R_bin)
    return _select_non_overlapping(candidates, top_k=30)


def apply_motif_correction(
    biased_notes: np.ndarray,
    motif_segments: list[dict],
    w_self: float = 0.6,
    w_motif: float = 0.4,
) -> np.ndarray:
    """モチーフ対応に基づいてノート確率を補正する。"""
    n_grids = biased_notes.shape[0]
    contributions = np.zeros_like(biased_notes)
    counts = np.zeros(n_grids, dtype=np.float64) # 正規化用のカウント

    for seg in motif_segments:
        a0 = seg["start_a"]
        b0 = seg["start_b"]
        L = seg["length"]
        valid_len = min(L, n_grids - a0, n_grids - b0)
        if valid_len <= 0:
            continue
        
        # 対応するモチーフ間でノートの確率を補完し合う
        a_sl = slice(a0, a0 + valid_len)
        b_sl = slice(b0, b0 + valid_len)

        contributions[a_sl] += biased_notes[b_sl]
        counts[a0:a0 + valid_len] += 1
        contributions[b_sl] += biased_notes[a_sl]
        counts[b0:b0 + valid_len] += 1

    has_motif = counts > 0
    corrected = biased_notes.copy()
    if has_motif.any():
        avg = contributions[has_motif] / counts[has_motif, np.newaxis]
        # 重み付き和で補正
        corrected[has_motif] = w_self * biased_notes[has_motif] + w_motif * avg

    return corrected
