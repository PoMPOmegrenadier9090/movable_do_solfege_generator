# solfege-gen
キーの変化と相対音名を考慮したソルフェージュの自動作成

## Flaskアプリ (同期処理)

アップロードした音声から、以下を同期処理で生成します。

- `vocals_transcribed.mid`
- `vocals_transcribed_with_inst.wav`
- `solfege.json` (推定キー・ノート・移動ド)

### 実行方法

1. 依存関係をインストール
2. `ffmpeg` をインストール (`brew install ffmpeg`)
3. アプリ起動

```bash
uv run main.py
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

1. Demucsで分離 (ボーカル・ベース・ドラムス・その他を生成)
2. basic-pitchを利用して，ボーカルを採譜するための特徴量作成
3. ビートを基準としたグリッド化により量子化
4. Viterbiアルゴリズムを利用し，グリッドのブロックごとにキー推定
5. キー情報，モチーフ反復を考慮した特徴量補正
6. 特徴量に基づいたMIDIデータ生成
7. 推定キーに基づく移動ド(ド/レ/ミ)ソルフェージュを生成

### 環境構築

#### 実行環境

- Python 3.9〜3.11 (uvを用いて環境構築します)
- macOS M4

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


## 参考文献
1. https://github.com/librosa/librosa
2. https://github.com/facebookresearch/demucs
3. https://github.com/nomadkaraoke/python-audio-separator
