FROM nvidia/cuda:12.8.1-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV VENV_PATH=/opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

ARG AITK_REPO=https://github.com/ArtDesignAwesome/ai-toolkit_BIG-DADDY-VERSION.git
ARG AITK_REF=main

RUN apt-get update && apt-get install --no-install-recommends -y \
    git \
    curl \
    build-essential \
    cmake \
    python3.12 \
    python3-pip \
    python3-dev \
    python3-venv \
    ffmpeg \
    tmux \
    htop \
    nvtop \
    python3-opencv \
    openssh-client \
    openssh-server \
    openssl \
    rsync \
    unzip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get update \
    && apt-get install --no-install-recommends -y nodejs \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv "${VENV_PATH}" \
    && python -m pip install --upgrade pip setuptools wheel

RUN git clone --depth 1 --branch "${AITK_REF}" "${AITK_REPO}" /app/ai-toolkit

WORKDIR /app/ai-toolkit

RUN python -m pip install --no-cache-dir torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu126 \
    && python -m pip install --no-cache-dir -r requirements.txt

WORKDIR /app/ai-toolkit/ui
RUN npm install \
    && npm run build \
    && npm run update_db

WORKDIR /
COPY start.sh /start.sh
RUN chmod +x /start.sh

EXPOSE 22 8675

CMD ["/start.sh"]
