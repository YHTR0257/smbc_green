# NVIDIAが提供するCUDA対応の公式ベースイメージを使用
# CUDAバージョンは、PyTorchが対応している最新の安定版に合わせるのが推奨されます。
# 現時点（2025年6月）のPyTorchの安定版はCUDA 12.1に対応しています。
# NVIDIAは常に最新のCUDAバージョンに対応するイメージを提供しています。
# 参考: https://hub.docker.com/r/nvidia/cuda/
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# 環境変数の設定
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHON_VERSION=3.11

# 必要なパッケージのインストール
RUN apt-get update && apt-get install -y --no-install-recommends \
    python${PYTHON_VERSION} \
    python3-pip \
    git \
    sudo \
    build-essential \
    curl \
    libffi-dev \
    openssh-client \
    rsync \
    && rm -rf /var/lib/apt/lists/*

# シンボリックリンクでPythonのデフォルトを更新
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VERSION} 1

# pipを最新にアップグレード
RUN pip install --no-cache-dir --upgrade pip

# 作業ディレクトリの設定
WORKDIR /workspace

# /etc/sudoers.d ディレクトリが存在しない場合に作成する
RUN mkdir -p /etc/sudoers.d
RUN pip install --no-cache-dir --upgrade pip

# vscode ユーザーを作成
RUN useradd -m -s /bin/bash vscode \
    && echo "vscode ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/vscode \
    && chmod 0440 /etc/sudoers.d/vscode

# Set up Python environment
ENV HOME=/home/vscode
USER vscode


# requirements.txtをコピーしてPython依存関係をインストール
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# PyTorchのインストール
# PyTorchの公式推奨インストール方法を使用（CUDA 12.1対応）
# --index-url を使用して、NVIDIA GPU対応のホイールファイルを指定
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# プロジェクトコードをコピー
COPY . .

# アプリケーションの実行コマンド (必要に応じて変更)
# CMD ["python", "src/main.py"] # 例