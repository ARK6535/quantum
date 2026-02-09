# VQE & 簡易分子動力学シミュレーション

水素分子 (H₂) を中心に、VQE (Variational Quantum Eigensolver) とその結果を使った簡易分子動力学シミュレーションを試すスクリプト群です。ノイズなしの statevector で動かす場合は IBM Quantum アカウントなしでも実行できますが、ノイズ込みシミュレーション (`vqe_h2.py`, `vqe_LiH.py`) は IBM Quantum アカウントが必要です。必要になったら https://quantum.cloud.ibm.com/docs/en/guides/cloud-setup を参照して登録してください。東大アカウントが存在するので研究室の人にも聞いてください。

このコードは好きにAIに渡していただいて結構です。最初は何やってるかよくわかんないと思うのでガンガン渡しましょう。

メモ: sto-3g は Pulay 補正が入りあんまり良くないらしいので、将来的にはより良い基底関数系に対応させるのが良いかもしれません。基底関数系は `--basis` オプションで変更できます。

H2のVQEはstatevectorの場合実行におよそ3秒、ノイズ込みの場合は500秒程度かかります。分子動力学シミュレーションの場合はそれにステップ数をかけた時間がかかります。

## セットアップ

リポジトリをクローンまたはフォークしてください。クローンだと権限周りで対応できるかわからないのでフォークが良いかとおもいます。最悪全ファイルをコピペしましょう。

推奨環境: PySCF まわりの制限により、macOS での実行を推奨します。Windows の場合 WSL でもおそらく動かせますが未確認です。Python バージョンは 3.11 です。

```zsh
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## モジュール構成

| ファイル | 役割 |
|---|---|
| `h2_helpers.py` | 共有定数、量子化学ユーティリティ、ログパーサ、`save_run_config()` |
| `h2_dynamics.py` | 分子動力学ヘルパー: Velocity-Verlet 積分、力の計算ラッパー、チェックポイント |
| `h2_energy.py` | ノイズ込み VQE エネルギー・力の計算 |
| `h2_energy_statevector.py` | ノイズなし statevector VQE エネルギー・力の計算 |
| `h2_energy_demo.py` | メインエントリポイント: MD シミュレーション実行 |
| `h2_energy_distribution.py` | 固定距離での VQE エネルギーの統計サンプリング |
| `vqe_h2.py` | H₂ 単発ノイズ込み VQE (IBM バックエンド必要) |
| `vqe_LiH.py` | LiH 単発ノイズ込み VQE (IBM バックエンド必要) |
| `plot_energy_vs_distance.py` | 可視化: ログから距離-エネルギー曲線を描画 |
| `plot_force_from_log.py` | 可視化: ログから距離-力の曲線を描画 |
| `read_csv.py` | 可視化: MD 軌道データ (CSV) からグラフを作成 |

## クイックスタート (分子動力学デモ)

```zsh
# デフォルト設定で1000ステップ実行 (statevector, ノイズなし)
python h2_energy_demo.py

# 短めのテスト実行
python h2_energy_demo.py --n-steps 10 --initial-distance 0.8

# 結果を可視化
python read_csv.py --log-dir logs/<YYMMDDHHmm>
```

1 行目で H₂ 振動シミュレーションを実行し、`logs/<YYMMDDHHmm>/` 配下に結果を保存します。ログディレクトリのパスは実行開始時にターミナルに表示されます。

### h2_energy_demo.py の主なオプション

| オプション | デフォルト | 説明 |
|---|---|---|
| `--initial-distance` | 0.8 | 初期 H-H 距離 (Å) |
| `--initial-velocity` | 0.0 | 初期相対速度 (Å/fs) |
| `--dt` | 0.01 | 時間刻み (fs) |
| `--n-steps` | 1000 | Velocity-Verlet ステップ数 |
| `--basis` | sto-3g | 基底関数系 |
| `--ansatz-reps` | 1 | アンザッツの繰り返し層数 |
| `--maxiter` | 2000 | オプティマイザの最大反復回数 (ステージあたり) |
| `--resume LOG_DIR` | — | 前回の実行を中断地点から再開 |
| `--verbose` | — | デバッグログを有効化 |

## チェックポイントと再開

分子動力学シミュレーションは長時間かかるため、途中で強制終了しても進捗を失わない仕組みがあります。

- 各ステップ終了時に `checkpoint.json` をアトミックに書き出します
- CSV (`dynamics_seq.csv`) もステップごとに逐次フラッシュします
- 中断後、`--resume` で同じログディレクトリを指定すると、最後のチェックポイントから計算を再開します

```zsh
# 長時間実行を開始
python h2_energy_demo.py --n-steps 500

# Ctrl+C で中断...

# 同じログディレクトリを指定して再開 (ステップ数は最終目標値)
python h2_energy_demo.py --resume logs/2602091701 --n-steps 500
```

再開時、CSV には前回分のデータが引き継がれ、以降のステップが追記されます。

## VQE 単体の実行

いずれも IBM Quantum のバックエンドノイズモデルを使ったノイズ込みシミュレーションです。

```zsh
# H₂ (デフォルト: 距離 0.735 Å, 1024 shots)
python vqe_h2.py
python vqe_h2.py --distance 0.9 --shots 4096 --basis sto-3g

# LiH (デフォルト: 距離 1.56 Å, 10000 shots)
python vqe_LiH.py
python vqe_LiH.py --distance 1.6 --basis sto-6g --ansatz-reps 3
```

### vqe_h2.py のオプション

| オプション | デフォルト | 説明 |
|---|---|---|
| `--distance` | 0.735 | H-H 距離 (Å) |
| `--basis` | sto-3g | 基底関数系 |
| `--ansatz-reps` | 0 | EfficientSU2 の繰り返し層数 |
| `--shots` | 1024 | 測定ショット数 |
| `--maxiter` | 300 | COBYLA の最大反復回数 |
| `--backend` | ibm_kawasaki | IBM バックエンド名 |

### vqe_LiH.py のオプション

| オプション | デフォルト | 説明 |
|---|---|---|
| `--distance` | 1.56 | Li-H 距離 (Å) |
| `--basis` | sto-6g | 基底関数系 |
| `--ansatz-reps` | 2 | EfficientSU2 の繰り返し層数 |
| `--shots` | 10000 | 測定ショット数 |
| `--cas-norb` | 5 | CASCI の活性空間軌道数 |
| `--cas-nelec` | 1 1 | CASCI の活性電子数 (α, β) |
| `--maxiter` | 300 | COBYLA の最大反復回数 |

## VQE エネルギー分布の統計サンプリング

```zsh
python h2_energy_distribution.py --distance 0.735 --samples 50 --workers 4
```

固定距離で VQE を複数回実行し、エネルギーのばらつきを評価します。

## 可視化

```zsh
# 距離-エネルギー曲線 (VQE ログから)
python plot_energy_vs_distance.py --log-dir logs/2511301524

# 距離-力の曲線
python plot_force_from_log.py --log-dir logs/2511301524

# MD 軌道データ (dynamics_seq.csv から)
python read_csv.py --log-dir logs/2602091701
```

## 実行パラメータの記録 (run_config.json)

すべてのエントリポイントは実行時に `run_config.json` をログディレクトリに保存します。再現性のため、以下の情報が記録されます。

- 実行コマンド (`sys.argv`)
- Python バージョン、OS
- 主要パッケージのバージョン (qiskit, qiskit_aer, pyscf, numpy, scipy)
- バックエンド種別 (statevector / noisy)
- 全 CLI 引数
- 関数デフォルト値 (cholesky_tol など、CLI に露出していないパラメータ)

## ログディレクトリの構成

```
logs/<YYMMDDHHmm>/
    run_config.json              # 実行パラメータ・環境情報
    checkpoint.json              # MD チェックポイント (再開用)
    dynamics_seq.csv             # MD 軌道データ
    h2_energy_quantum_0.80.txt   # VQE トレースログ (距離ごと)
    _energy_vs_distance_*.txt    # エネルギーまとめテーブル
    force_vs_distance_vqe.txt    # 力のまとめテーブル
```

- タイムスタンプ形式: `YYMMDDHHmm` (例: `2602091345`)
- VQE 実行ごとに TXT が書き出されますが、同じ時刻に近い原子間距離を複数回走らせると上書きされます。分子動力学シミュレーションでは距離変化が小さいため一部の VQE ログが欠損しますが、シミュレーション結果 (CSV) には影響しません。

## Git-ignored ファイル

`apikey.json`, `.env`, `logs/`, `*.png`, `*.pdf`, `__pycache__/`, `.venv/` などはリポジトリに含まれません。詳細は `.gitignore` を参照してください。