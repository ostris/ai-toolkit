# Setup script for AI Toolkit on Windows
# This script sets up the environment, installs dependencies, and validates the installation

$ErrorActionPreference = "Stop"

function Write-Info {
    Write-Host "[INFO] $args" -ForegroundColor Blue
}

function Write-Success {
    Write-Host "[SUCCESS] $args" -ForegroundColor Green
}

function Write-Warning {
    Write-Host "[WARNING] $args" -ForegroundColor Yellow
}

function Write-Error {
    Write-Host "[ERROR] $args" -ForegroundColor Red
}

Write-Info "AI Toolkit Setup Script"
Write-Info "======================"
Write-Host ""

# Check if we're in the right directory
if (-not (Test-Path "run.py") -or -not (Test-Path "requirements.txt")) {
    Write-Error "Please run this script from the ai-toolkit root directory"
    exit 1
}

# Check for Python
try {
    $pythonVersion = python --version 2>&1
    Write-Success "Python detected: $pythonVersion"
} catch {
    Write-Error "Python is not installed or not in PATH. Please install Python 3.10 or higher."
    exit 1
}

# Check Python version
$versionOutput = python --version 2>&1
$versionMatch = $versionOutput -match "Python (\d+)\.(\d+)"
if ($versionMatch) {
    $major = [int]$matches[1]
    $minor = [int]$matches[2]
    if ($major -lt 3 -or ($major -eq 3 -and $minor -lt 10)) {
        Write-Error "Python 3.10 or higher is required. Found: $major.$minor"
        exit 1
    }
}

# Create virtual environment if it doesn't exist
if (Test-Path ".venv") {
    Write-Info "Virtual environment already exists (.venv)"
    $pythonCmd = ".venv\Scripts\python.exe"
    & .venv\Scripts\Activate.ps1
} elseif (Test-Path "venv") {
    Write-Info "Virtual environment already exists (venv)"
    $pythonCmd = "venv\Scripts\python.exe"
    & venv\Scripts\Activate.ps1
} else {
    Write-Info "Creating virtual environment..."
    python -m venv venv
    $pythonCmd = "venv\Scripts\python.exe"
    & venv\Scripts\Activate.ps1
    Write-Success "Virtual environment created"
}

# Upgrade pip
Write-Info "Upgrading pip..."
& $pythonCmd -m pip install --upgrade pip

# Detect GPU backend
Write-Info "Detecting GPU backend..."
$hasRocm = $false
try {
    $null = Get-Command rocm-smi -ErrorAction Stop
    $hasRocm = $true
} catch {
    # Check for NVIDIA
    try {
        $null = nvidia-smi
        Write-Info "NVIDIA GPU detected. Installing PyTorch with CUDA support..."
    } catch {
        Write-Info "No GPU detected or drivers not installed. Installing PyTorch with CUDA support (CPU fallback)..."
    }
}

if ($hasRocm) {
    Write-Info "ROCm detected. Installing PyTorch with ROCm support..."
    
    # Function to map GPU architecture to ROCm directory name
    function Map-GpuArchToRocmDir {
        param([string]$detectedArch)
        
        # Map gfx110X variants to gfx110X-all (RDNA 3 architecture)
        if ($detectedArch -match "gfx110[0-3]") {
            return "gfx110X-all"
        }
        
        # Direct mappings for architectures that match their directory names
        switch ($detectedArch) {
            { $_ -in @("gfx1151", "gfx1030", "gfx90a", "gfx906", "gfx908", "gfx941", "gfx942") } {
                return $detectedArch
            }
            default {
                # For unknown architectures, try the detected name first
                return $detectedArch
            }
        }
    }
    
    # Get GPU architecture
    if (-not $env:PYTORCH_ROCM_ARCH) {
        Write-Info "Detecting GPU architecture..."
        try {
            $gpuInfo = rocm-smi --showproductname 2>&1 | Out-String
            
            # Extract gfx architecture code from rocm-smi output
            $detectedArch = ""
            if ($gpuInfo -match "gfx\d+") {
                $detectedArch = $matches[0]
            }
            
            if ($detectedArch) {
                $rocmArch = Map-GpuArchToRocmDir $detectedArch
                Write-Info "Detected GPU architecture: $detectedArch"
                Write-Info "Mapped to ROCm directory: $rocmArch"
                $env:PYTORCH_ROCM_ARCH = $rocmArch
            } else {
                Write-Warning "Could not auto-detect GPU architecture from rocm-smi output"
                Write-Info "Common architectures:"
                Write-Info "  - gfx1151 (RDNA 3.5 - Strix Point Halo APU)"
                Write-Info "  - gfx110X-all (RDNA 3 - RX 7900/7800/7700 series, gfx1100/gfx1101/gfx1102/gfx1103)"
                Write-Info "  - gfx1030 (RDNA 2 - RX 6900/6800/6700 series)"
                Write-Info "  - gfx90a (CDNA 2 - Instinct MI200 series)"
                $userInput = Read-Host "Enter your GPU architecture [gfx1151]"
                $env:PYTORCH_ROCM_ARCH = if ($userInput) { $userInput } else { "gfx1151" }
            }
        } catch {
            Write-Warning "Failed to detect GPU architecture: $_"
            Write-Info "Common architectures:"
            Write-Info "  - gfx1151 (RDNA 3.5 - Strix Point Halo APU)"
            Write-Info "  - gfx110X-all (RDNA 3 - RX 7900/7800/7700 series)"
            Write-Info "  - gfx1030 (RDNA 2 - RX 6900/6800/6700 series)"
            $userInput = Read-Host "Enter your GPU architecture [gfx1151]"
            $env:PYTORCH_ROCM_ARCH = if ($userInput) { $userInput } else { "gfx1151" }
        }
        
        Write-Info "Using GPU architecture: $env:PYTORCH_ROCM_ARCH"
        
        # Install PyTorch with ROCm
        Write-Info "Installing PyTorch with ROCm support..."
        $indexUrl = "https://rocm.nightlies.amd.com/v2/$($env:PYTORCH_ROCM_ARCH)/"
        & $pythonCmd -m pip install --upgrade --index-url $indexUrl --pre torch torchaudio torchvision
    } else {
        Write-Info "Using GPU architecture from environment: $env:PYTORCH_ROCM_ARCH"
        $indexUrl = "https://rocm.nightlies.amd.com/v2/$($env:PYTORCH_ROCM_ARCH)/"
        & $pythonCmd -m pip install --upgrade --index-url $indexUrl --pre torch torchaudio torchvision
    }
} else {
    Write-Info "Installing PyTorch with CUDA support..."
    & $pythonCmd -m pip install --no-cache-dir torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu126
}

# Install other dependencies
Write-Info "Installing other dependencies from requirements.txt..."
& $pythonCmd -m pip install -r requirements.txt

# Verify installation
Write-Info "Verifying installation..."
try {
    & $pythonCmd -c "import torch"
    Write-Success "PyTorch installed successfully"
    
    # Check GPU availability
    $gpuCheck = & $pythonCmd -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>&1
    if ($LASTEXITCODE -eq 0) {
        $deviceName = & $pythonCmd -c "import torch; print(torch.cuda.get_device_name(0))" 2>&1
        Write-Success "GPU detected: $deviceName"
    } else {
        Write-Warning "No GPU detected. Training will run on CPU (very slow)."
    }
} catch {
    Write-Error "PyTorch installation verification failed: $_"
    exit 1
}

# Check other critical dependencies
$missingDeps = @()
$deps = @("accelerate", "diffusers", "transformers")
foreach ($dep in $deps) {
    try {
        & $pythonCmd -c "import $dep" 2>&1 | Out-Null
    } catch {
        $missingDeps += $dep
    }
}

if ($missingDeps.Count -gt 0) {
    Write-Warning "Some dependencies are missing: $($missingDeps -join ', ')"
    Write-Info "Try running: pip install -r requirements.txt"
} else {
    Write-Success "All core dependencies verified"
}

Write-Success "Setup complete!"
Write-Info "You can now run the toolkit with: .\start_toolkit.ps1 ui"
Write-Info "Or start training with: .\start_toolkit.ps1 train config\your_config.yaml"

