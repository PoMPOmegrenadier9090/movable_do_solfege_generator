# solfege-gen
ソルフェージュを自動作成したい

## Flaskアプリ (同期処理)

アップロードした音声から、以下を同期処理で生成します。

- `vocals_hmm.mid`
- `vocals_hmm_with_inst.wav`
- `solfege.json` (推定キー・ノート・移動ド)

### 実行方法

1. 依存関係をインストール
2. `ffmpeg` をインストール
3. アプリ起動

```bash
python main.py
```

4. ブラウザで `http://localhost:2026` を開く

### ジョブ保存方式

ジョブはクライアント単位のディレクトリ配下で管理します。

```text
runtime/
	clients/
		<client_id>/
			jobs/
				<job_id>/
					uploads/
					artifacts/
```

リクエストごとに `job_id` を発行し、成果物はジョブ単位で分離保存します。

### パイプライン

1. Demucsで分離 (`vocals` を利用)
2. UVR_MDXNET_KARAで分離 (`Instrumental` を利用)
3. UVR-DeEcho-DeReverbで `vocals` から No Reverb を生成
4. basic-pitchベースでMIDI生成
5. MIDI音源を伴奏とミックス
6. 推定キーに基づく移動ド(ド/レ/ミ)を生成

### 環境構築

#### 前提条件

- Python 3.9〜3.11
- macOS (Apple Silicon) の場合、以下のセットアップが必要

#### Apple Silicon (ARM64) での libsamplerate セットアップ

`audio-separator` が依存する `samplerate` パッケージには x86_64 用の `libsamplerate.dylib` が同梱されており、Apple Silicon 環境では読み込めません。Homebrew で ARM64 ネイティブ版をインストールし、置き換えます。

```bash
# 1. Homebrew で libsamplerate をインストール
brew install libsamplerate

# 2. パッケージ同梱の dylib を ARM64 版で置き換え
cp $(brew --prefix libsamplerate)/lib/libsamplerate.dylib \
  .venv/lib/python3.11/site-packages/samplerate/_samplerate_data/libsamplerate.dylib
```

> **注意**: `.venv` 再作成や `samplerate` パッケージ更新のたびに再実行が必要です。

#### demucsによるパート分離
- 公式リポジトリのminimal requirementsをインストール
- ffmpegのインストール
- ipykernel, demucsのインストール

#### torchCREPEによるボーカルのMIDI変換
basic-pitchよりは，うまく捉えることができている．
ただ，以下のような課題もある．

- リズムがぐちゃぐちゃ
- 細かい音，特にダイアトニック以外の音を拾うのが苦手そう