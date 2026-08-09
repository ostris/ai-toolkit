# AI Toolkit Repository Summary

## プロジェクト概要

AI Toolkit by Ostrisは、diffusionモデルの包括的な学習スイートです。最新のモデルをコンシューマーグレードのハードウェアで動作させることを目指しており、画像・動画モデルの両方をサポートしています。GUI・CLI両方のインターフェースを提供し、使いやすさと機能の豊富さを両立しています。

## 主要な特徴

### サポートモデル

- **FLUX.1 (dev/schnell)**: 最新のdiffusionモデル（24GB VRAM必要）
- **SD 1.5/SDXL**: Stable Diffusion系モデル
- **その他**: Lumina、Chroma、HiDream、OmniGen2、Wan2.1など

### 学習手法

- **LoRA (Low-Rank Adaptation)**: 効率的なfine-tuning
- **LoKr**: 高度なadaptation手法
- **Full Fine-tuning**: 完全なモデル学習
- **Slider Training**: 特定の属性制御

### インターフェース

- **CLI**: コマンドライン経由での実行
- **WebUI**: ブラウザベースのインターフェース (ui/ フォルダ)
- **Gradio UI**: シンプルなGradio-based UI (flux_train_ui.py)

## ディレクトリ構造

```bash
ai-toolkit/
├── config/                    # 設定ファイル
│   └── examples/              # 各種学習設定の例
├── extensions/                # 拡張機能
├── extensions_built_in/       # 内蔵拡張機能
├── jobs/                      # ジョブ処理システム
├── toolkit/                   # コア機能
│   ├── models/                # モデル実装
│   ├── optimizers/            # オプティマイザー
│   └── samplers/              # サンプラー
├── ui/                        # WebUIフロントエンド
└── scripts/                   # ユーティリティスクリプト
```

## 重要ファイル

### エントリーポイント

- `run.py`: メインの実行ファイル
- `main.py`: シンプルなHello World
- `flux_train_ui.py`: Gradio UIのエントリーポイント

### コア機能

- `toolkit/job.py`: ジョブ管理システム
- `toolkit/config.py`: 設定管理
- `toolkit/data_loader.py`: データ処理
- `toolkit/stable_diffusion_model.py`: SD系モデル実装

### 設定例

- `config/examples/train_lora_flux_24gb.yaml`: FLUX.1 LoRA学習設定
- `config/examples/train_lora_sd35_large_24gb.yaml`: SD 3.5学習設定

## 学習フロー

### 1. データセット準備

```bash
dataset/
├── image1.jpg
├── image1.txt
├── image2.jpg
└── image2.txt
```

### 2. 設定ファイル作成

```yaml
job: extension
config:
  name: "my_model"
  process:
    - type: 'sd_trainer'
      training_folder: "output"
      network:
        type: "lora"
        linear: 16
        linear_alpha: 16
      datasets:
        - folder_path: "/path/to/dataset"
          resolution: [512, 768, 1024]
      train:
        batch_size: 1
        steps: 2000
        lr: 1e-4
```

### 3. 学習実行

```bash
python run.py config/my_config.yaml
```

## 技術的特徴

### GPU要件

- **FLUX.1**: 24GB VRAM以上
- **SDXL**: 12GB VRAM以上
- **SD 1.5**: 8GB VRAM以上

### 最適化機能

- **Gradient Checkpointing**: メモリ効率化
- **Mixed Precision**: 高速化
- **Dynamic Batching**: 効率的なバッチ処理
- **Quantization**: モデル圧縮

### 拡張システム

- プラグインベースの拡張機能
- カスタムプロセッサー対応
- Hook システム

## Cloud環境サポート

### Modal

- `run_modal.py`: Modal環境での実行
- `config/examples/modal/`: Modal用設定例

### Beam Cloud

- `run_beam.py`: Beam CloudのGPU FunctionとGPU付きAI Toolkit GUI Podを提供
- `docker/Dockerfile.beam`: CUDA 12.9、学習依存関係、Next.js UIを含むBeamイメージ
- `config/examples/beam/`: Beam用設定例
- `BEAM_GPU`でServerless GPUを明示選択し、`BEAM_POOL`で予約済みOn-demand GPUを指定
- GUI Podのアイドル停止時間は`BEAM_KEEP_WARM_SECONDS`で指定（既定1800秒、`-1`は無期限）
- On-demandマシンはTTL付きで予約し、学習完了後にCLIで解放
- GUIのSQLite、データセット、学習結果、モデルキャッシュはBeam Volumeに永続化

### RunPod

- 詳細なRunPod設定手順をREADMEに記載

### Docker

- `docker/Dockerfile`: コンテナイメージ
- `docker-compose.yml`: 簡単なデプロイ

## WebUI機能

### 主要機能

- ジョブ管理・監視
- データセット管理
- リアルタイム学習進捗
- サンプル画像生成・表示
- GPU監視
- 設定管理

### 技術スタック

- **Frontend**: Next.js + TypeScript
- **Backend**: Node.js API
- **Database**: Prisma + SQLite
- **UI**: Tailwind CSS

## 依存関係

### Python環境

- Python >= 3.10
- PyTorch 2.7.0
- Transformers 4.52.4
- Diffusers (カスタムフォーク)
- Accelerate, PEFT, Bitsandbytes

### 特別な依存関係

- `lycoris-lora`: LoKr実装
- `optimum-quanto`: 量子化
- `controlnet_aux`: ControlNet機能

## 最近の更新

### 2025年7月

- WebUIの動画モデル設定改善
- Wan I2V学習対応

### 2025年6月

- FLUX.1 Kontext学習サポート
- OmniGen2対応
- 教示データセット学習

## 開発・コントリビューション

### ブランチ構造

- `main`: メインブランチ
- `feat/modal`: Modal機能開発

### 拡張開発

- `extensions/`: カスタム拡張
- `extensions_built_in/`: 内蔵拡張

### テスト

- `testing/`: テストスクリプト
- `notebooks/`: Jupyter notebook例

## 商用・ライセンス

### FLUX.1-dev

- 非商用ライセンス
- HuggingFace認証必要

### FLUX.1-schnell

- Apache 2.0ライセンス
- 商用利用可能

## スポンサー・サポート

活発なコミュニティサポートがあり、GitHub Sponsors、Patreon、PayPalでの支援を受け付けています。大手企業（Replicate、Hugging Face等）からのスポンサードも受けています。

## 使用例

### 基本的なLoRA学習

```bash
# 設定ファイルコピー
cp config/examples/train_lora_flux_24gb.yaml config/my_flux_lora.yaml

# 設定編集後実行
python run.py config/my_flux_lora.yaml
```

### WebUI起動

```bash
cd ui
npm run build_and_start
# http://localhost:8675 でアクセス
```

このリポジトリは、AI研究者・開発者・アーティストが最新のdiffusionモデルを効率的に学習・カスタマイズできる包括的なツールセットを提供しています。
