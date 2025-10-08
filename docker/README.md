# Docker Build and Deployment Guide

This guide covers building and deploying the AI Toolkit API server as a Docker container.

## Prerequisites

- Docker installed with GPU support (NVIDIA Container Toolkit)
- Access to push to `ghcr.io/civitai` (GitHub Container Registry)
- GitHub Personal Access Token with `write:packages` scope

## Authentication

Before pushing images, authenticate with GitHub Container Registry:

```bash
# Login to GHCR
echo $GITHUB_TOKEN | docker login ghcr.io -u USERNAME --password-stdin
```

Replace `$GITHUB_TOKEN` with your GitHub Personal Access Token and `USERNAME` with your GitHub username.

## Building the API Server Image

```bash
# Build the image
docker build -f docker/Dockerfile.api -t ai-toolkit-api .

# Tag for GHCR
docker tag ai-toolkit-api ghcr.io/civitai/civitai-ai-toolkit:latest

# Push to registry
docker push ghcr.io/civitai/civitai-ai-toolkit:latest
```

## Running the API Server

Pull and run from registry:

```bash
docker pull ghcr.io/civitai/civitai-ai-toolkit:latest

docker run -d \
  --name ai-toolkit-api \
  --gpus all \
  -p 8000:8000 \
  -v ./output:/app/ai-toolkit/output \
  -v ./models:/app/ai-toolkit/models \
  ghcr.io/civitai/civitai-ai-toolkit:latest
```

## Managing the Container

```bash
# View logs
docker logs ai-toolkit-api -f

# Stop container
docker stop ai-toolkit-api

# Start container
docker start ai-toolkit-api

# Restart container
docker restart ai-toolkit-api

# Remove container
docker rm -f ai-toolkit-api

# Check container status
docker ps --filter "name=ai-toolkit-api"
```

## Dockerfile Comparison

### `Dockerfile` (Original - UI Server)
- Clones code from GitHub repository
- Installs Node.js and builds the UI frontend
- Exposes port 8675
- Runs the UI server with `npm run start`

### `Dockerfile.api` (API Server)
- Uses local code (no git clone)
- Skips Node.js and UI build steps
- Includes API-specific dependencies (libglib2.0-0, libsm6, etc.)
- Exposes port 8000
- Runs FastAPI server with `uvicorn api_server.app:app`

## Versioning

To create versioned releases:

```bash
# Tag with version
docker tag ai-toolkit-api ghcr.io/civitai/civitai-ai-toolkit:v1.0.0
docker tag ai-toolkit-api ghcr.io/civitai/civitai-ai-toolkit:latest

# Push both tags
docker push ghcr.io/civitai/civitai-ai-toolkit:v1.0.0
docker push ghcr.io/civitai/civitai-ai-toolkit:latest
```

## Image Size

The built image is approximately **41GB** and includes:
- CUDA 12.8.1 runtime
- PyTorch nightly build with CUDA support
- All Python dependencies from requirements.txt
- API server code

## Troubleshooting

### Build fails with permission errors
Ensure you have write permissions to the docker directory and sufficient disk space.

### Push fails with authentication error
Re-authenticate with GitHub Container Registry:
```bash
docker logout ghcr.io
echo $GITHUB_TOKEN | docker login ghcr.io -u USERNAME --password-stdin
```

### Container won't start with GPU
Ensure NVIDIA Container Toolkit is installed:
```bash
# Check NVIDIA runtime is available
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu22.04 nvidia-smi
```

## API Documentation

Once running, the API documentation is available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`
