$ErrorActionPreference = "Stop"

$RootDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $RootDir

$PythonBin = if ($env:PYTHON_BIN) { $env:PYTHON_BIN } else { "python" }
$VenvDir = if ($env:VENV_DIR) { $env:VENV_DIR } else { "venv" }

& $PythonBin -m venv $VenvDir

$ActivateScript = Join-Path $VenvDir "Scripts\Activate.ps1"
& $ActivateScript

python -m pip install --upgrade pip
python -m pip install --no-cache-dir torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt

Write-Host ""
Write-Host "Bootstrap complete."
Write-Host "Activate with: .\$VenvDir\Scripts\Activate.ps1"
