# AI Toolkit by Ostris

## DGX OS installation instructions

You need to use Python 3.11 to run AI Toolkit on DGX OS. The easiest way to do this without affecting the system installation of Python is to create a virtual environment with **miniconda**, which allows you to specify the version of Python to use in the environment.

This guide will assume you have a fresh installation of DGX OS, and will guide you through the installation of all requirements.

### Installation instructions for DGX OS:

**1) Get Python 3.11 (via miniconda)**

Install the latest version of miniconda:
```
$ wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh
$ chmod u+x Miniconda3-latest-Linux-aarch64.sh
$ ./Miniconda3-latest-Linux-aarch64.sh
```

Restart your bash or ssh session. If miniconda was installed successfully, it will automatically load the 'base' environment by default. If you want to disable this behaviour, run:
```
$ conda config --set auto_activate_base false
```

Now you can create a Python 3.11 environment for ai-toolkit:
```
$ conda create --name ai-toolkit python=3.11
```

Then activate the environment with:

```
$ conda activate ai-toolkit
```


**2) Install PyTorch**

```
$ pip3 install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu130
```


**3) Install the remaining requirements (dgx_requirements.txt)**

```
$ pip3 install -r dgx_requirements.txt
```

### Running the UI on DGX OS:

Running the UI is not that different from doing it on other systems, however, you need to install the ARM64 version of NodeJS for Linux, which is compatible with the NVIDIA Grace CPU.


**1) Install Node.js**

Download a Linux ARM64 build of Node.js from: https://nodejs.org (for example: https://nodejs.org/dist/v24.11.1/node-v24.11.1-linux-arm64.tar.xz)

Extract it and add the bin directory to your path. I extracted it to **/opt** and added the following to my ~/.bashrc file:
```
export PATH=“/opt/node-v24.11.1-linux-arm64/bin:$PATH”
```


**2) Compile and run the Node.js UI**

Change to the ui directory, then build and run the UI:
```
$ cd ui
$ npm run build_and_start
```

If all went well, you’ll be able to access the UI on port 8675 and start training.


<details>
  <summary>Troubleshooting issues</summary>
If you’re not getting any output when starting a training job from the UI, it’s probably crashing before the process started, the best way to debug these issues is to run the python training script directly (which is normally started by the UI). To do this, set up a training job in the UI, go to the advanced config screen, copy and paste the configuration into a file like train.yaml, then run the training script like this with the conda virtual environment active:

```
$ python run.py path/to/train.yaml
```
</details>
<br>