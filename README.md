# solfege-gen
ソルフェージュを自動作成したい

### 環境構築
#### demucsによるパート分離
- 公式リポジトリのminimal requirementsをインストール
- ffmpegのインストール
- ipykernel, demucsのインストール

#### torchCREPEによるボーカルのMIDI変換
basic-pitchよりは，うまく捉えることができている．
ただ，以下のような課題もある．

- リズムがぐちゃぐちゃ
- 細かい音，特にダイアトニック以外の音を拾うのが苦手そう