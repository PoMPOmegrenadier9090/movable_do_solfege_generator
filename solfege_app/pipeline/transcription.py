"""採譜パイプライン。

ノートブック (midi_transcription.ipynb) と同等の処理を行う:
  1. チューニング補正 + リサンプリング
  2. HPSS → ハーモニック成分抽出
  3. テンポ・ビート検出
  4. basic-pitch 推論
  5. ビートグリッド量子化
  6. キー推定 (クロマ + HMM Viterbi)
  7. スケールバイアス適用
  8. グリーディノート割り当て
  9. モチーフ抽出・補正
  10. 動的 Viterbi HMM デコード
  11. MIDI 合成 + 伴奏ミックス
  12. ソルフェージュ生成
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pretty_midi
import librosa
import numpy as np
import soundfile as sf
from basic_pitch import ICASSP_2022_MODEL_PATH
from basic_pitch.inference import predict

from .beat_grid import (
    detect_beats,
    quantize_to_grid,
    center_weighted_agg_func
)
from .key_estimation import (
    KeyEstimate,
    apply_scale_bias,
    estimate_key_sequence,
)
from .motif import apply_motif_correction, extract_motifs
from .note_assignment import (
    greedy_note_assignment,
    note_list_to_midi
)
from .separation import DemucsSeparationResult
from .solfege import attach_solfege
from .. import config

logger = logging.getLogger(__name__)


def run_transcription(
    demucs_result: DemucsSeparationResult,
    artifacts_dir: Path,
    original_audio_path: Path | None = None,
) -> dict:
    """フル採譜パイプラインを実行する。

    Parameters
    ----------
    vocal_no_reverb_path : リバーブ除去済みボーカル
    instrumental_path : UVR 分離済みインストゥルメンタル
    artifacts_dir : 成果物の保存先ディレクトリ
    original_audio_path : 元の音源 (キー推定用、省略時は instrumental を使用)
    """
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    # --- 0. Demucsによる分離結果を読み込み ---
    y_vocals_raw, sr = librosa.load(str(demucs_result.vocals))
    if sr != config.SAMPLE_RATE:
        raise ValueError(f"Sample rate of vocals is {sr}, expected {config.SAMPLE_RATE}")
    y_bass_raw, _ = librosa.load(str(demucs_result.bass))
    y_drums_raw, _ = librosa.load(str(demucs_result.drums))
    y_other_raw, _ = librosa.load(str(demucs_result.other))
    # インスト音源の合成
    min_len = min(len(y_vocals_raw), len(y_bass_raw), len(y_drums_raw), len(y_other_raw))
    y_vocals_raw = y_vocals_raw[:min_len]
    y_bass_raw = y_bass_raw[:min_len]
    y_drums_raw = y_drums_raw[:min_len]
    y_other_raw = y_other_raw[:min_len]
    y_inst_raw = y_bass_raw + y_drums_raw + y_other_raw
    del y_bass_raw, y_drums_raw, y_other_raw

    # ── 1. チューニング補正 ──
    logger.info("Step 1: Tuning correction")
    tuning_offset = librosa.estimate_tuning(y=y_inst_raw, sr=sr)
    tuning_rate = 2 ** (-tuning_offset / 12)
    # チューニング補正: orig_sr * tuning_rate でリサンプリングすることで、
    # 実質的に再生速度を tuning_rate 倍にする（サンプリングレートは sr のまま）
    adjusted_sr = sr * tuning_rate
    y_inst = librosa.resample(y_inst_raw, orig_sr=adjusted_sr, target_sr=sr)
    y_vocals = librosa.resample(y_vocals_raw, orig_sr=adjusted_sr, target_sr=sr)
    logger.info(f"  tuning_offset={tuning_offset:.3f}, tuning_rate={tuning_rate:.4f}, sr={sr}")
    # キー推定後に不要になった変数を削除
    del y_inst_raw, y_vocals_raw

    # ── 2. ボーカル HPSS ──
    logger.info("Step 2: HPSS on vocals")
    y_harmonic, _ = librosa.effects.hpss(y_vocals, margin=1.0)
    harmonic_path = artifacts_dir / "vocals_harmonic.wav"
    sf.write(str(harmonic_path), y_harmonic, sr)
    del y_vocals, y_harmonic

    # ── 3. テンポ・ビート検出 ──
    logger.info("Step 3: Beat detection")
    midi_tempo, beat_frames = detect_beats(y_inst, sr)
    beat_times_sec = librosa.frames_to_time(beat_frames, sr=sr, hop_length=config.HOP_LENGTH)
    logger.info(f"  tempo={midi_tempo:.1f} BPM, beats={len(beat_frames)}")

    # ── 4. basic-pitch 推論 ──
    logger.info("Step 4: basic-pitch inference")
    model_output, basic_pitch_midi, _ = predict(
        str(harmonic_path),
        model_or_model_path=ICASSP_2022_MODEL_PATH,
        onset_threshold=0.5,
        frame_threshold=0.3,
        minimum_note_length=20,
        maximum_frequency=2000,
        multiple_pitch_bends=False,
        melodia_trick=True,
        midi_tempo=midi_tempo,
    )
    onsets_np = model_output['onset']  # shape: (num_frames, num_pitches)
    pitches_np = model_output['note']  # shape: (num_frames, num_pitches)
    # basic_pitchのframeの実測SPFを計算
    # モデルが MIDI 変換したファイル
    pm_basic = basic_pitch_midi
    notes_basic = pm_basic.instruments[0].notes

    spf = config.SPF_THEORETICAL
    if len(notes_basic) > 0:
        # MIDIの最後のノートから実測SPFを逆算
        last_note = notes_basic[-1]
        last_pitch_idx = last_note.pitch - config.MIDI_OFFSET
        # 最後のノートの音高を持つフレームを取得
        last_frames = np.where(onsets_np[:, last_pitch_idx] > 0.5)[0]
        
        if len(last_frames) > 0:
            last_frame = last_frames[-1]
            # frameとMIDI時間の対応よりsecond per frameを計算
            spf_actual = last_note.start / last_frame
            logger.info(f"  Auto-calibrated SPF: {spf_actual:.10f} s/frame")
            spf = spf_actual
        else:
            # フォールバック：理論値を使用
            logger.info("  Actual SPF couldnt be calculated")

    # ── 5. ビートグリッド量子化 ──
    logger.info("Step 5: Beat grid quantization")
    grid_onsets_np, grid_times_np = quantize_to_grid(
        onsets_np, 
        beat_times_sec, 
        spf, 
        config.SUBDIVISIONS, 
        agg_func=np.max)
    grid_notes_np, _ = quantize_to_grid(
        pitches_np, 
        beat_times_sec, 
        spf, 
        config.SUBDIVISIONS, 
        agg_func=center_weighted_agg_func)
    logger.info(f"  grids={grid_onsets_np.shape[0]}, pitches={grid_onsets_np.shape[1]}")

    # ── 6. キー推定 ──
    logger.info("Step 6: Key estimation")
    y_for_key = y_inst
    if original_audio_path is not None:
        y_for_key, _ = librosa.load(str(original_audio_path), sr=sr)
        y_for_key = librosa.resample(y_for_key, orig_sr=adjusted_sr, target_sr=sr)
    key_sequence, global_key = estimate_key_sequence(
        y_for_key, sr, spf, beat_times_sec, block_size=config.SUBDIVISIONS
    )
    logger.info(f"  global_key={global_key.label}")

    # ── 7. スケールバイアス適用 ──
    logger.info("Step 7: Scale bias")
    biased_notes, biased_onsets = apply_scale_bias(grid_notes_np, grid_onsets_np, key_sequence)

    # ── 8. モチーフ抽出・補正 ──
    logger.info("Step 8: Motif extraction & correction")
    motif_segments = extract_motifs(biased_notes)
    biased_notes = apply_motif_correction(biased_notes, motif_segments)
    logger.info(f"  motifs={len(motif_segments)}")

    # ── 9. グリーディノート割り当て (スケールバイアス + モチーフ補正済み) ──
    logger.info("Step 9: Greedy note assignment (motif-corrected)")
    assigned_pitch, notes = greedy_note_assignment(
        biased_onsets, 
        biased_notes, 
        grid_times_np, 
        pitch_threshold=0.40,
    )

    # ── 10. MIDI 保存 + 伴奏ミックス ──
    logger.info("Step 10: MIDI save & mix")
    pm = note_list_to_midi(notes)
    midi_path = artifacts_dir / "vocals_transcribed.mid"
    pm.write(str(midi_path))

    y_transcribed = pm.synthesize(fs=int(sr)).astype(np.float32)
    mix_len = max(len(y_inst), len(y_transcribed))
    y_inst_pad = np.pad(y_inst.astype(np.float32), (0, mix_len - len(y_inst)))
    y_transcribed_pad = np.pad(y_transcribed, (0, mix_len - len(y_transcribed)))

    mix_audio = y_inst_pad + 0.5 * y_transcribed_pad
    peak = np.max(np.abs(mix_audio))
    if peak > 1.0:
        mix_audio = mix_audio / peak

    mix_path = artifacts_dir / "vocals_transcribed_with_inst.wav"
    sf.write(str(mix_path), mix_audio, sr)

    # ── 11. ソルフェージュ生成 ──
    logger.info("Step 11: Solfege generation")
    # Create note_events from pretty_midi.Note objects
    note_events: list[dict] = []
    for note in sorted(notes, key=lambda n: (n.start, n.pitch)):
        note_events.append({
            "start": float(note.start),
            "end": float(note.end),
            "pitch": int(note.pitch),
            "velocity": int(note.velocity),
        })
    note_events_with_solfege = attach_solfege(note_events, key_sequence, grid_times_np)

    # key_sequence 情報をリストに変換（JSON シリアライズ用）
    from .solfege import key_index_to_label
    key_sequence_info = [
        {"grid_time": float(t), "key": key_index_to_label(int(k))}
        for t, k in zip(grid_times_np, key_sequence)
    ]

    solfege_json_path = artifacts_dir / "solfege.json"
    with solfege_json_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "estimated_global_key": global_key.label,
                "key_sequence": key_sequence_info,
                "note_count": len(note_events_with_solfege),
                "notes": note_events_with_solfege,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    logger.info("Transcription pipeline complete")
    return {
        "sample_rate": sr,
        "estimated_global_key": global_key.label,
        "midi_filename": midi_path.name,
        "mix_filename": mix_path.name,
        "solfege_filename": solfege_json_path.name,
        "note_count": len(note_events_with_solfege),
        "notes": note_events_with_solfege,
        "key_sequence": key_sequence_info,
    }
