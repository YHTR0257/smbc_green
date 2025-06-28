# PyTorch公式イメージを使用（動作確認済み: CUDA 11.8）
FROM pytorch/pytorch:2.7.1-cuda11.8-cudnn9-runtime

# 環境変数の設定
ENV DEBIAN_FRONTEND=noninteractive

# GPU最適化のための環境変数（念のため）
ENV CUDA_LAUNCH_BLOCKING=1
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# 追加のシステムパッケージをインストール
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    sudo \
    build-essential \
    curl \
    libffi-dev \
    openssh-client \
    rsync \
    vim \
    wget \
    && rm -rf /var/lib/apt/lists/*

# /etc/sudoers.d ディレクトリが存在しない場合に作成
RUN mkdir -p /etc/sudoers.d

# vscode ユーザーを作成（VS Code Dev Containersとの互換性のため）
RUN useradd -m -s /bin/bash vscode \
    && echo "vscode ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/vscode \
    && chmod 0440 /etc/sudoers.d/vscode

# 作業ディレクトリの設定
WORKDIR /workspace

# requirements.txtを最初にコピー（Dockerレイヤーキャッシュ最適化）
COPY requirements.txt .

# pip cache更新とrequirements.txtからインストール
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# プロジェクトコードをコピー
COPY . .

# デバイス検出テストスクリプトをコピー（プロジェクトにtest_device.pyとして保存）
# COPY test_device.py .

# ユーザー環境の設定
ENV HOME=/home/vscode
USER vscode

# デフォルトコマンド（デバイステストを実行）
CMD ["python", "test_device.py"]