# Getting Started with AI-Toolkit

## Connect to Dheyo01 H100 

```bash
ssh dheyo01@lh100.dheyo.ai
```

## Launch tmux Session - Optional[Useful for longer runs]

```bash
tmux a -t shivanvitha
```

If it doesn't exist, use the below:

```bash
tmux new -s shivanvitha
```

## Navigate to the Workdir and Activate Environment 

```bash
cd /data/shivanvitha/dheyo_ai_toolkit/ai-toolkit
```

```bash
source ../../ai-toolkit/aitool/bin/activate
```

## Launch UI

```bash
cd ui
```

```bash
npm run build_and_start
```

## Live Monitor H100

```bash
watch -n0.1 nvidia-smi
```


## Checkout the UI on Port 7777 on Browser

```
http://99.50.230.172:7777 
```

OR 

```
lh100.dheyo.ai:7777
```


## Add Dataset(s)
Coming soon with images..

## Create a New Job
Coming soon with images..
