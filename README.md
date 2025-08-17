# AI-Toolkit with AMD MI300X GPUs

## Set up and Installations

### Clone the Repo
```bash
git clone git@github.com:dheyoai/ai-toolkit.git
```

### Switch to AMD branch
```bash
git checkout dheyo_amd
```

### Create and Activate Virtual Environment

```bash
python3 -m venv aitool
source aitool/bin/activate
```

### Install pytorch

```bash
uv pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm6.4
```

### Install Other Dependencies

```bash
pip install -r requirements.txt
```

## CLI Launch
```bash
python3 run.py <path/to/config/yaml>
```

## GUI Launch

```bash
cd ui
npm run build_and_start
```

Open localhost/loopback URL on port 7777 on the browser
```
http://localhost:7777/
```

**Note**: Make sure to use SSH Tunneling when using gpu-22 or gpu-60 
```bash
ssh -L 7777:localhost:7777 ubuntu@gpu-22
```

## MI300X Monitoring Dashboard
![AMD MI300X Dashboard](./assets/gpu_dashboard.png)

## Notes

Support for the following features has been disabled temporarily 
- Quantization of DiT/Text Encoder with Torchao
- 8-bit Optimizer Quantization using bitsandbytes

Therefore, you cannot use quantization on AMD GPUs for now. 