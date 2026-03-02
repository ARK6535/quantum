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

## 関数の呼び出しフロー

各シミュレーションで実行される主要な関数の流れを示します。

### 共通の量子化学パイプライン (h2_helpers.py)

全てのシミュレーションで共通して呼ばれるハミルトニアン構築の流れです。与えられた原子間距離と基底関数系から、量子コンピュータ上で扱える 4-qubit のハミルトニアンを生成します。

まず PySCF で水素分子の Mole オブジェクトを構築し、制限 Hartree-Fock (RHF) 計算で分子軌道を求めます。得られた分子軌道のうち結合性軌道と反結合性軌道の 2 本を活性空間として選び、CASCI(2,2) で対角化します。STO-3G 基底の場合、分子軌道が 2 本しかないため CAS(2,2) は完全 CI と一致します。

活性空間の 1 電子積分と 2 電子積分を取り出した後、Jordan-Wigner 変換でフェルミオン演算子をパウリ演算子に変換します。2 電子積分については Cholesky 分解で低ランク近似し、パウリ項の数を抑えます。最終的に Qiskit の `SparsePauliOp` として 4-qubit ハミルトニアンが得られます。

```
_build_h2_qubit_hamiltonian(distance, basis)
 ├─ _build_h2_molecule()        # PySCF の gto.Mole を構築
 ├─ scf.RHF → kernel()          # Hartree-Fock 計算
 ├─ mcscf.CASCI(ncas=2, nelecas=(1,1))
 │    └─ sort_mo() → kernel()   # 活性空間 CAS(2,2) を対角化
 ├─ get_h1eff(), get_h2eff()    # 活性空間の 1e/2e 積分を取得
 └─ _build_sparse_pauli_hamiltonian(ecore, h1e, h2e)
      ├─ _creators_destructors()  # Jordan-Wigner 生成・消滅演算子
      ├─ _cholesky()               # 2e 積分の Cholesky 分解
      └─ SparsePauliOp を組み立て  # 4-qubit ハミルトニアン
```

### 分子動力学シミュレーション (h2_energy_demo.py)

H₂ の核間振動を Velocity-Verlet 法でシミュレーションします。各時間ステップで VQE を実行して、その幾何構造でのエネルギーと力を求め、原子核の位置と速度を更新します。

`main()` は CLI 引数の解析と `run_config.json` の保存を行った後、`dynamics_seq()` に処理を委譲します。`dynamics_seq()` は MD ループの本体で、ステップごとに `_eval_energy_force()` を呼びます。この内部関数が `--backend-type` に応じて statevector 版か noisy 版の VQE を実行し、エネルギーと Hellmann-Feynman 力のペアを返します。

statevector 版 (`compute_h2_energy_quantum_statevector`) はノイズなしの `AerSimulator(method='statevector')` を使い、UCCSD ansatz で COBYLA と L-BFGS-B の 2 段階最適化を行います。noisy 版 (`compute_h2_energy_quantum_noisy`) は IBM の実機ノイズモデルを載せた `AerSimulator.from_backend()` を使い、shots ベースの期待値評価を行います。noisy 版では `_build_pass_manager()` で動的デカップリングを含むトランスパイルも行います。

CSV はステップごとに flush し、checkpoint.json はアトミック書き込みするため、途中で kill しても直前ステップまでの結果が残ります。

```
main()
 ├─ parse_args()
 ├─ save_run_config()                          # 実行パラメータを JSON に記録
 ├─ energy_classical_bohr()                    # 参照用の古典エネルギー計算
 │    └─ compute_h2_energy_classical()         # (h2_helpers) RHF → CASCI(2,2)
 └─ dynamics_seq()                             # (h2_dynamics) Velocity-Verlet MD ループ
      │
      │  ┌─ [各ステップで呼ばれる] ─────────────────────────────┐
      ├─ _eval_energy_force(r_bohr)                              │
      │    │                                                     │
      │    ├─ [statevector の場合]                                │
      │    │   └─ compute_h2_energy_quantum_statevector()         │
      │    │        ├─ _build_h2_qubit_hamiltonian()              │
      │    │        │    ├─ _build_h2_molecule()     # PySCF Mole │
      │    │        │    ├─ RHF → CASCI(2,2)                     │
      │    │        │    └─ _build_sparse_pauli_hamiltonian()     │
      │    │        │         └─ _cholesky() + Jordan-Wigner 変換 │
      │    │        ├─ _build_h2_force_operator()    # dH/dR 演算子│
      │    │        ├─ UCCSD ansatz + HF reference 構築           │
      │    │        ├─ COBYLA → L-BFGS-B (2段階最適化)            │
      │    │        │    └─ BackendEstimatorV2 (statevector)      │
      │    │        └─ Hellmann-Feynman 力の期待値計算             │
      │    │                                                     │
      │    └─ [noisy の場合]                                     │
      │        └─ compute_h2_energy_quantum_noisy()              │
      │             ├─ (上記と同じハミルトニアン構築)              │
      │             ├─ AerSimulator.from_backend()  # ノイズモデル│
      │             ├─ _build_pass_manager() → トランスパイル     │
      │             ├─ COBYLA → L-BFGS-B                         │
      │             │    └─ BackendEstimatorV2 (noisy, shots指定) │
      │             └─ Hellmann-Feynman 力の期待値計算             │
      │                                                          │
      ├─ Velocity-Verlet 積分 (r, v を更新)                      │
      ├─ CSV に 1 行追記 + flush                                 │
      └─ checkpoint.json をアトミック書き込み                     │
           └─────────────────────────────────────────────────────┘
```

### VQE 単体実行 (vqe_h2.py)

特定の原子間距離で 1 回だけノイズ込み VQE を実行し、エネルギーの収束過程をプロットします。IBM Quantum の実機バックエンドからノイズモデルを取得し、`AerSimulator.from_backend()` でローカルにノイズ込みシミュレーションを行います。

MD 用の VQE とは異なり、ansatz は `EfficientSU2` (ハードウェア効率型) を使い、最適化は COBYLA の 1 段階のみです。ハミルトニアン構築には `h2_helpers.py` の共通パイプラインを使います。トランスパイル時に動的デカップリング (DD) を挿入し、idle 時間中のデコヒーレンスを抑制します。

```
main()
 ├─ parse_args()
 ├─ _build_h2_qubit_hamiltonian()              # (h2_helpers) ハミルトニアン構築
 ├─ save_run_config()
 ├─ QiskitRuntimeService → backend 取得
 ├─ EfficientSU2 ansatz 構築
 ├─ generate_preset_pass_manager() → トランスパイル (DD 付き)
 ├─ AerSimulator.from_backend()                # ノイズモデル付きシミュレータ
 ├─ COBYLA 最適化ループ
 │    └─ _cost_func() → BackendEstimatorV2
 └─ 結果プロット
```

### VQE エネルギー分布 (h2_energy_distribution.py)

固定距離で VQE を繰り返し実行し、得られるエネルギーの統計分布を評価します。VQE は初期パラメータや最適化経路の違いにより毎回わずかに異なる結果を返すため、そのばらつきを定量化します。

`ProcessPoolExecutor` でワーカーを並列起動し、各ワーカーが `run_batch()` 内で指定回数の VQE を逐次実行します。個々の VQE は statevector 版を使います。

```
main()
 ├─ parse_args()
 ├─ save_run_config()
 └─ ProcessPoolExecutor で並列実行
      └─ run_batch()  ×n_workers
           └─ compute_h2_energy_quantum_statevector()  ×n_samples
                └─ (上記 statevector フローと同じ)
```

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