# PyTorch公式イメージを使用（CUDA 12.8対応の最新devel版）
FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel

# 環境変数の設定
ENV DEBIAN_FRONTEND=noninteractive

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

# ユーザー環境の設定
ENV HOME=/home/vscode
USER vscode

# デフォルトコマンド
CMD ["bash"]