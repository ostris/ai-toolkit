## Docker :
### Build :
  - ```cd .```
  - ```docker build --no-cache -f docker/Dockerfile -t ai-toolkit:0 .```

### Run :
  - ```docker run --rm --gpus '"device=0"'  -e PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:2048 -it --network host ai-toolkit:0```