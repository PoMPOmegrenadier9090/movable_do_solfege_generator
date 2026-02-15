"""キー推定モジュール。

クロマベクトル + HMM Viterbi により時間変化するキー系列を推定する。
"""
from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np
from scipy.ndimage import gaussian_filter1d

from .beat_grid import HOP_LENGTH, quantize_to_grid
from .. import config




@dataclass
class KeyEstimate:
    """推定キー情報。"""
    root_pc: int       # 0–11
    mode: str          # "major" | "minor"
    label: str         # e.g. "C Major"
    key_index: int     # 0–23 (0–11: Major, 12–23: Minor)


def _build_key_templates() -> np.ndarray:
    """24キーの正規化済みテンプレート (24, 12)。"""
    templates_major = np.array([np.roll(config.MAJOR_PROFILE, i) for i in range(12)])
    templates_minor = np.array([np.roll(config.MINOR_PROFILE, i) for i in range(12)])
    key_templates = np.vstack([templates_major, templates_minor])
    centered = key_templates - key_templates.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(centered, axis=1, keepdims=True) + 1e-10
    return centered / norms


def _build_transition_matrix(n_states: int = 24) -> np.ndarray:
    """キー遷移行列を構築する。"""
    STABLE_PROB = config.KEY_ESTIMATION_HMM_PARAMS["STABLE_PROB"]
    RELATIVE_PROB = config.KEY_ESTIMATION_HMM_PARAMS["RELATIVE_PROB"]
    FIFTH_PROB = config.KEY_ESTIMATION_HMM_PARAMS["FIFTH_PROB"]
    PARALLEL_PROB = config.KEY_ESTIMATION_HMM_PARAMS["PARALLEL_PROB"]

    trans = np.full((n_states, n_states), 0.001)

    for i in range(12):
        # Major key (i)
        trans[i, i] = STABLE_PROB # 自己遷移
        trans[i, (i + 7) % 12] = FIFTH_PROB # 属調
        trans[i, (i + 5) % 12] = FIFTH_PROB # 下属調
        trans[i, 12 + (i - 3) % 12] = RELATIVE_PROB # 平行調
        trans[i, 12 + i] = PARALLEL_PROB # 同主調

        # Minor key (12+i)
        m = 12 + i
        trans[m, m] = STABLE_PROB # 自己遷移
        trans[m, 12 + (i + 7) % 12] = FIFTH_PROB # 属調
        trans[m, 12 + (i + 5) % 12] = FIFTH_PROB # 下属調
        trans[m, (i + 3) % 12] = RELATIVE_PROB # 平行調
        trans[m, i] = PARALLEL_PROB # 同主調

    trans /= trans.sum(axis=1, keepdims=True)
    return trans


def estimate_key_sequence(
    y_original: np.ndarray,
    sr: int,
    spf: float,
    beat_times_sec: np.ndarray,
    block_size: int = 4,
    filter_sigma: float = 4.0,
) -> tuple[np.ndarray, KeyEstimate]:
    """時間変化するキー系列を推定する。

    Parameters
    ----------
    y_original : 元の音源（チューニング補正前）
    sr : サンプリングレート
    spf : 1フレームあたりの秒数
    beat_times_sec : ビートタイム配列
    block_size : HMM ブロックサイズ（グリッド数）
    filter_sigma : ガウシアンフィルタの sigma

    Returns
    -------
    key_sequence : (num_grids,) — 各グリッドのキーインデックス (0–23)
    global_key : 曲全体の推定キー
    """
    y_harmonic, _ = librosa.effects.hpss(y_original, margin=1.0)

    # --- クロマ特徴量の獲得 ---
    chroma = librosa.feature.chroma_cqt(
        y=y_harmonic, 
        sr=sr, 
        hop_length=config.HOP_LENGTH,
        fmin=librosa.note_to_hz("C1"), 
        n_octaves=config.N_OCTAVES
    ).T  # (num_frames, 12)

    # ビートグリッドに量子化
    grid_chroma, _ = quantize_to_grid(
        chroma, 
        beat_times_sec,
        spf, 
        config.SUBDIVISIONS, 
        agg_func=np.mean)

    # ガウシアン平滑化
    grid_chroma_filtered = gaussian_filter1d(
        grid_chroma, 
        sigma=filter_sigma, 
        axis=0, 
        mode="nearest"
    )

    # --- ブロック化 ---
    original_length = grid_chroma_filtered.shape[0]
    pad_width = (block_size - (original_length % block_size)) % block_size
    padded = np.pad(grid_chroma_filtered, ((0, pad_width), (0, 0)), mode="edge")
    # ブロック(block_size=4 => 1 拍)ごとに集約する
    coarse_chroma = padded.reshape(-1, block_size, 12).mean(axis=1) # shape: (num_blocks, 12)

    # --- キー推定 ---
    # ピアソン相関を用いて，各ブロックとキーの類似度を推定する
    template_norm = _build_key_templates() # (24, 12)
    chroma_centered = coarse_chroma - coarse_chroma.mean(axis=1, keepdims=True) # (num_blocks, 12)
    chroma_norm = chroma_centered / (np.linalg.norm(chroma_centered, axis=1, keepdims=True) + 1e-10) # (num_blocks, 12)
    correlation = chroma_norm @ template_norm.T # (num_blocks, 24)

    # グローバルキーの推定
    total_corr = correlation.sum(axis=0)
    global_key_idx = int(np.argmax(total_corr))

    # 確率変換 (Soft-max)
    key_prob = np.exp(correlation)
    key_prob /= key_prob.sum(axis=1, keepdims=True)

    # --- Viterbiアルゴリズムによるキー系列の推定 ---
    transition = _build_transition_matrix(n_states=24) # (24, 24)
    p_init = np.ones(24) * 0.1
    # グローバルキーの初期確率を補正
    p_init[global_key_idx] = 0.30
    p_init /= p_init.sum()
    # Viterbiアルゴリズム実行
    key_seq_coarse = librosa.sequence.viterbi(key_prob.T, transition, p_init=p_init)

    # 元の解像度に復元
    key_sequence = np.repeat(key_seq_coarse, block_size)[:original_length]

    # グローバルキー
    root_pc = global_key_idx % 12
    mode = "major" if global_key_idx < 12 else "minor"
    mode_label = "Major" if mode == "major" else "Minor"
    global_key = KeyEstimate(
        root_pc=root_pc,
        mode=mode,
        label=f"{config.KEY_NAMES[root_pc]} {mode_label}",
        key_index=global_key_idx,
    )

    return key_sequence, global_key


def get_scale_mask(
    key_idx: int, n_pitches: int = 88, midi_offset: int = 21,
) -> np.ndarray:
    """キーのスケール構成音に基づく重みマスクを返す。"""
    root = key_idx % 12 # 主音
    is_minor = key_idx >= 12

    if is_minor:
        intervals = {0, 2, 3, 5, 7, 8, 10} # Natural Minor
    else:
        intervals = {0, 2, 4, 5, 7, 9, 11} # Major

    mask = np.ones(n_pitches)
    for i in range(n_pitches):
        midi_note = i + midi_offset
        pitch_class = midi_note % 12
        relative_pitch = (pitch_class - root + 12) % 12

        if relative_pitch in intervals:
            mask[i] = 1.2 # スケール構成音を優遇
        elif is_minor and relative_pitch in {6, 11}:
            mask[i] = 0.9 # メロディックマイナー特徴音
        else:
            mask[i] = 0.6 # スケール外の音を抑制

    return mask


def apply_scale_bias(
    grid_notes: np.ndarray,
    grid_onsets: np.ndarray,
    key_sequence: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """キー系列に基づくスケールバイアスをノート確率に適用する。"""
    n_grids, n_pitches = grid_notes.shape

    # key_sequence の長さ合わせ
    if len(key_sequence) < n_grids:
        key_seq = np.pad(key_sequence, (0, n_grids - len(key_sequence)), mode="edge")
    elif len(key_sequence) > n_grids:
        key_seq = key_sequence[:n_grids]
    else:
        key_seq = key_sequence

    biased_notes = grid_notes.copy()
    biased_onsets = grid_onsets.copy()
    for t in range(n_grids):
        mask = get_scale_mask(int(key_seq[t]), n_pitches, config.MIDI_OFFSET)
        biased_notes[t] *= mask
        biased_onsets[t] *= mask

    return biased_notes, biased_onsets
