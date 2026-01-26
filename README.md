概要

水素分子 (H2) を中心に、VQE とその結果を使った簡易分子動力学を試すスクリプト群です。ノイズなしの statevector で動かす場合は IBM Quantum アカウントなしでも実行できますが、一部のコードはアカウント情報が必要です。必要になったら https://quantum.cloud.ibm.com/docs/en/guides/cloud-setup を参照して登録してください。

このコードは好きにAIに渡していただいて結構です。最初は何やってるかよくわかんないと思うのでガンガン渡しましょう。

セットアップ
- 推奨環境: pyscfまわりの制限により、macでの実行を推奨します。Windowsの場合、WSLでもおそらく動かせますが未確認です。 Pythonバージョンは3.11です。仮想環境を有効化してから実行してください。
	```zsh
	source .venv/bin/activate
	pip install -r requirements.txt
	```

よく使うスクリプト
- h2_energy_demo.py: H2 の簡易分子動力学シミュレーションを実行します。
- vqe_h2.py / vqe_LiH.py: H2 / LiH の ノイズありVQE を1回実行します。
- plot_energy_vs_distance.py: 指定日時の VQE ログから距離-エネルギー曲線を描画します。
- plot_force_from_log.py: 指定日時のログから距離-力の曲線を描画します。
- read_csv.py: 分子動力学計算結果を読み込み、グラフを作成します。

クイックスタート (分子動力学デモ)
```zsh
python h2_energy_demo.py
python read_csv.py
```
1 行目で H2 振動シミュレーションを実行し、`logs/` 配下に結果を保存します。2 行目で保存結果を読み込み、グラフを生成します。

VQE 単体の実行
- H2: `python vqe_h2.py`
- LiH: `python vqe_LiH.py` (PySCF が必要)

ログと上書きについて
- VQE 実行ごとに TXT が書き出されますが、同じ原子間距離を複数回走らせると上書きされます。
- `logs/<datetime>/` に距離ごとのエネルギーや付随データが入ります。`plot_energy_vs_distance.py` と `plot_force_from_log.py` はこのフォルダをまとめて可視化します。