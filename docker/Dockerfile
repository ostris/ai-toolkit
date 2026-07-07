FROM docker.io/nvidia/cuda:12.8.1-devel-ubuntu24.04

LABEL authors="jaret"

# Set noninteractive to avoid timezone prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/app/.venv/bin:/home/app/.local/bin:$PATH"

# ref https://en.wikipedia.org/wiki/CUDA
ENV TORCH_CUDA_ARCH_LIST="8.0 8.6 8.9 9.0 10.0 12.0"

# Install dependencies
RUN apt-get update && apt-get install --no-install-recommends -y \
    git \
    curl \
    build-essential \
    cmake \
    wget \
    python3.12 \
    python3-pip \
    python3-dev \
    python3-setuptools \
    python3-wheel \
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

# Install nodejs
WORKDIR /tmp
RUN curl -sL https://deb.nodesource.com/setup_23.x -o nodesource_setup.sh && \
    bash nodesource_setup.sh && \
    apt-get update && \
    apt-get install -y nodejs && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN useradd -m app
RUN mkdir -p /app && chown -R app:app /app

COPY --chown=app:app ui/package.json ui/package-lock.json /app/ui/
COPY --chown=app:app requirements.txt requirements_base.txt /app/
COPY --chown=app:app docker/start.sh /app/start.sh

RUN touch /etc/rp_environment && chown app:app /etc/rp_environment

USER app

WORKDIR /app

RUN python3.12 -m venv .venv

# install pytorch before cache bust to avoid redownloading pytorch
RUN /app/.venv/bin/python -m pip install --no-cache-dir torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu128

# ---------------------------------------------------------------------------- #
# Dependency layers come BEFORE the source clone so they are only rebuilt (and
# only need to be re-pulled by servers) when the dependency manifests change,
# not on every code change.
# ---------------------------------------------------------------------------- #

# Install Python dependencies (only re-runs when the requirements files change)
RUN /app/.venv/bin/python -m pip install --no-cache-dir -r requirements.txt && \
    /app/.venv/bin/python -m pip install --no-cache-dir  setuptools==69.5.1

# Install Node dependencies (only re-runs when package.json / package-lock.json change)
RUN cd /app/ui && npm ci

# ---------------------------------------------------------------------------- #
# Source code comes LAST. Only this layer (plus the UI build below) is rebuilt
# on a code change, so servers only re-pull the (small) source, not the deps.
# Clone to a temp dir and rsync the source in, preserving the dependency dirs
# already populated above (ui/node_modules) and the manifests already used.
# ---------------------------------------------------------------------------- #
ARG CACHEBUST=1234
ARG GIT_COMMIT=main
RUN echo "Cache bust: ${CACHEBUST}" && \
    git clone https://github.com/ostris/ai-toolkit.git /tmp/ai-toolkit-src && \
    cd /tmp/ai-toolkit-src && \
    git checkout ${GIT_COMMIT} && \
    rsync -a --delete \
        --exclude 'start.sh' \
        --exclude 'ui/node_modules' \
        --exclude 'requirements.txt' \
        --exclude 'ui/package.json' \
        --exclude 'ui/package-lock.json' \
        /tmp/ai-toolkit-src/ /app/ && \
    rm -rf /tmp/ai-toolkit-src

# Build UI (re-runs on code change, but reuses the cached node_modules above).
# update_db runs first because it does `prisma generate`, which creates the
# @prisma/client types the TS build needs. In the old layout generate happened
# as a side effect of npm install seeing the schema; now the source arrives
# after npm ci, so run it explicitly before the build.
RUN cd /app/ui && \
    npm run update_db && \
    npm run build

# Expose port (assuming the application runs on port 3000)
EXPOSE 8675

RUN chmod +x /app/start.sh

CMD ["/app/start.sh"]