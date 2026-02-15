from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from demucs.separate import main as demucs_separator
from audio_separator.separator import Separator

def _find_first(paths: Iterable[Path], predicate) -> Path | None:
    """
    パスのリストから，条件に合致する最初のファイルを見つけて返す．
    条件に合致するファイルがない場合は None を返す．
    """
    for path in paths:
        if predicate(path):
            return path
    return None

@dataclass
class DemucsSeparationResult:
    """htdemucsの出力結果を格納するデータクラス"""
    vocals: Path
    drums: Path
    bass: Path
    other: Path


def run_separation_by_demucs(input_audio: Path, output_root: Path) -> DemucsSeparationResult:
    output_root.mkdir(parents=True, exist_ok=True)
    # コマンドライン引数を構成
    args = ["-o", str(output_root), "-n", "htdemucs", "--mp3", str(input_audio)]
    # コマンドライン引数を指定して実行
    demucs_separator(args)
    # Demucsの出力先ディレクトリ下の，ファイル名以下のパスを探索する
    # output_root/htdemucs/{input_audio_stem}/vocals.mp3 などを想定
    audio_stem = input_audio.stem
    
    # 各ステムのファイルを探索
    stems = {}
    for stem_name in ["vocals", "drums", "bass", "other"]:
        candidates = list(output_root.rglob(f"{stem_name}.mp3")) + list(output_root.rglob(f"{stem_name}.wav"))
        exact = _find_first(candidates, lambda p: audio_stem in str(p.parent))
        if exact:
            stems[stem_name] = exact
        elif candidates:
            stems[stem_name] = candidates[0]
        else:
            raise FileNotFoundError(f"demucs {stem_name} output not found")
    
    return DemucsSeparationResult(
        vocals=stems["vocals"],
        drums=stems["drums"],
        bass=stems["bass"],
        other=stems["other"],
    )


def _as_path_list(outputs: list[str], cwd: Path) -> list[Path]:
    """
    出力されたパスを絶対パスに変換する．
    すでに絶対パスである場合はそのまま返す．
    """
    result: list[Path] = []
    for entry in outputs:
        p = Path(entry)
        if not p.is_absolute():
            p = (cwd / p).resolve()
        result.append(p)
    return result



@dataclass
class SeparationArtifacts:
    input_audio: Path
    demucs_vocals: Path
    demucs_drums: Path
    demucs_bass: Path
    demucs_other: Path
    uvr_vocals: Path
    uvr_instrumental: Path
    dereverb_vocal: Path


class AudioSeparationPipeline:
    """音声分離処理と関連パスを一元管理するクラス。"""

    def __init__(self, input_audio: Path, job_dir: Path) -> None:
        self.input_audio = Path(input_audio)
        self.job_dir = Path(job_dir)

        self.processed_dir = self.job_dir / "processed"
        self.demucs_dir = self.processed_dir / "htdemucs"
        self.uvr_work_dir = self.processed_dir / "uvr"

        self.demucs_dir.mkdir(parents=True, exist_ok=True)
        self.uvr_work_dir.mkdir(parents=True, exist_ok=True)

        self.demucs_vocals: Path | None = None
        self.demucs_drums: Path | None = None
        self.demucs_bass: Path | None = None
        self.demucs_other: Path | None = None
        self.uvr_vocals: Path | None = None
        self.uvr_instrumental: Path | None = None
        self.dereverb_vocal: Path | None = None
    
    def _pick_uvr_output(self, paths: list[Path], keyword: str) -> Path:
        """
        paths内にある，keywordを含むファイル名のパスを返す．
        """
        keyword_l = keyword.lower()
        for path in paths:
            if keyword_l in path.name.lower():
                return path
        raise FileNotFoundError(f"UVR output with keyword '{keyword}' not found")

    def run_demucs(self) -> DemucsSeparationResult:
        """
        Demucsによる音源分離（vocals, drums, bass, other）
        """
        result = run_separation_by_demucs(self.input_audio, self.demucs_dir)
        self.demucs_vocals = result.vocals
        self.demucs_drums = result.drums
        self.demucs_bass = result.bass
        self.demucs_other = result.other
        return result

    def run_instrumental_separation(self) -> tuple[Path, Path]:
        """
        UVRを用いてインスト音源とボーカルの分離を行う．
        """
        separator = Separator(output_dir=str(self.uvr_work_dir))
        separator.load_model("UVR_MDXNET_KARA.onnx")
        outputs = separator.separate(str(self.input_audio))
        output_paths = _as_path_list(outputs, self.uvr_work_dir)
        
        print(f"DEBUG: UVR_MDXNET_KARA outputs: {[str(p) for p in output_paths]}")

        self.uvr_instrumental = self._pick_uvr_output(output_paths, "instrumental")
        self.uvr_vocals = self._pick_uvr_output(output_paths, "vocals")
        return self.uvr_vocals, self.uvr_instrumental

    def run_echo_and_reverb_removal(self, vocals_path: Path | None = None) -> Path:
        """
        UVR-DeEcho-DeReverbを用いて，ボーカルからエコーとリバーブを除去する．
        """
        target_vocals = vocals_path or self.demucs_vocals
        if target_vocals is None:
            raise ValueError("demucs_vocals is not set. Run run_demucs() first or pass vocals_path.")

        separator = Separator(output_dir=str(self.uvr_work_dir))
        separator.load_model("UVR-DeEcho-DeReverb.pth")
        outputs = separator.separate(str(target_vocals))
        output_paths = _as_path_list(outputs, self.uvr_work_dir)
        
        print(f"DEBUG: UVR-DeEcho-DeReverb outputs: {[str(p) for p in output_paths]}")

        self.dereverb_vocal = self._pick_uvr_output(output_paths, "no reverb")
        return self.dereverb_vocal

    def run_all(self) -> SeparationArtifacts:
        self.run_demucs()
        self.run_instrumental_separation()
        self.run_echo_and_reverb_removal()

        if (self.demucs_vocals is None or self.demucs_drums is None or 
            self.demucs_bass is None or self.demucs_other is None or
            self.uvr_vocals is None or self.uvr_instrumental is None or 
            self.dereverb_vocal is None):
            raise RuntimeError("Separation outputs are incomplete")

        return SeparationArtifacts(
            input_audio=self.input_audio,
            demucs_vocals=self.demucs_vocals,
            demucs_drums=self.demucs_drums,
            demucs_bass=self.demucs_bass,
            demucs_other=self.demucs_other,
            uvr_vocals=self.uvr_vocals,
            uvr_instrumental=self.uvr_instrumental,
            dereverb_vocal=self.dereverb_vocal,
        )
