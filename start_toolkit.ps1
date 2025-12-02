# Startup script for AI Toolkit on Windows
# This script sets up the environment and launches the toolkit

$ErrorActionPreference = "Stop"

# Colors for output
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

# Default values
$MODE = "help"
$CONFIG_FILE = ""
$RECOVER = $false
$JOB_NAME = ""
$LOG_FILE = ""
$UI_PORT = 8675
$UI_DEV_MODE = $false

# Function to show usage
function Show-Usage {
    Write-Host "AI Toolkit Startup Script"
    Write-Host "=========================="
    Write-Host ""
    Write-Host "Usage: .\start_toolkit.ps1 [MODE] [OPTIONS]"
    Write-Host ""
    Write-Host "Modes:"
    Write-Host "  setup                                    - Setup/validate the toolkit environment"
    Write-Host "  train <config_file> [config_file2 ...]  - Run training job(s) with config file(s)"
    Write-Host "  gradio                                  - Launch Gradio UI for FLUX training"
    Write-Host "  ui                                       - Launch web UI (Next.js, production mode)"
    Write-Host "  ui --dev                                 - Launch web UI (development mode with hot reload)"
    Write-Host "  help                                     - Show this help message"
    Write-Host ""
    Write-Host "Training Options:"
    Write-Host "  -r, --recover                            - Continue running additional jobs if one fails"
    Write-Host "  -n, --name NAME                          - Name to replace [name] tag in config"
    Write-Host "  -l, --log FILE                           - Log file to write output to"
    Write-Host ""
    Write-Host "UI Options:"
    Write-Host "  -p, --port PORT                          - Port for web UI (default: 8675)"
    Write-Host ""
    Write-Host "Examples:"
    Write-Host "  .\start_toolkit.ps1 train config\examples\train_lora_wan22_14b_24gb.yaml"
    Write-Host "  .\start_toolkit.ps1 train config\my_config.yaml -r -n my_training_run"
    Write-Host "  .\start_toolkit.ps1 gradio"
    Write-Host "  .\start_toolkit.ps1 ui"
    Write-Host ""
}

# Function to get Python executable path
function Get-PythonCmd {
    # If we're in a virtual environment, use its Python
    if ($env:VIRTUAL_ENV) {
        return "$env:VIRTUAL_ENV\Scripts\python.exe"
    }
    
    # Check for uv venv
    if ((Test-Path ".venv") -and (Test-Path ".venv\Scripts\python.exe")) {
        return ".venv\Scripts\python.exe"
    }
    
    # Check for standard venv
    if ((Test-Path "venv") -and (Test-Path "venv\Scripts\python.exe")) {
        return "venv\Scripts\python.exe"
    }
    
    # Fall back to system python
    return "python"
}

# Function to check and activate virtual environment
function Setup-Venv {
    # Check if we're already in a virtual environment
    if ($env:VIRTUAL_ENV) {
        Write-Info "Virtual environment already active: $env:VIRTUAL_ENV"
        return
    }
    
    # Check for uv venv
    if (Test-Path ".venv") {
        Write-Info "Activating uv virtual environment..."
        & .venv\Scripts\Activate.ps1
        Write-Success "Virtual environment activated"
        return
    }
    
    # Check for standard venv
    if (Test-Path "venv") {
        Write-Info "Activating virtual environment..."
        & venv\Scripts\Activate.ps1
        Write-Success "Virtual environment activated"
        return
    }
    
    Write-Warning "No virtual environment found. Using system Python."
    Write-Info "Consider creating one with: python -m venv venv"
}

# Function to detect backend and set ROCm environment variables
function Detect-Backend {
    Write-Info "Detecting GPU backend..."
    
    $pythonCmd = Get-PythonCmd
    
    try {
        $backendOutput = & $pythonCmd -c "import torch; print('ROCm' if hasattr(torch.version, 'hip') and torch.version.hip else 'CUDA')" 2>&1
        if ($LASTEXITCODE -eq 0) {
            $script:BACKEND = $backendOutput
            Write-Success "Backend detected: $BACKEND"
            
            if ($BACKEND -eq "ROCm") {
                # Set ROCm environment variables (Windows ROCm support is limited)
                Write-Info "ROCm detected. Note: ROCm on Windows has limited support."
                Write-Info "Consider using CUDA on Windows for better compatibility."
            }
            
            # Verify GPU is available
            $gpuCheck = & $pythonCmd -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>&1
            if ($LASTEXITCODE -eq 0) {
                $deviceName = & $pythonCmd -c "import torch; print(torch.cuda.get_device_name(0))" 2>&1
                Write-Success "GPU detected: $deviceName"
            } else {
                Write-Warning "No GPU detected. Training will run on CPU (very slow)."
            }
        } else {
            Write-Warning "PyTorch not available. Make sure it's installed."
            $script:BACKEND = "UNKNOWN"
        }
    } catch {
        Write-Warning "PyTorch not available. Make sure it's installed."
        $script:BACKEND = "UNKNOWN"
    }
}

# Function to verify dependencies
function Verify-Dependencies {
    Write-Info "Verifying dependencies..."
    
    $pythonCmd = Get-PythonCmd
    
    try {
        & $pythonCmd -c "import torch" 2>&1 | Out-Null
        if ($LASTEXITCODE -ne 0) {
            Write-Error "PyTorch is not installed!"
            Write-Info "Run: .\setup.ps1 to install dependencies"
            Write-Info "Or manually: pip install torch torchvision torchaudio"
            return $false
        }
    } catch {
        Write-Error "PyTorch is not installed!"
        Write-Info "Run: .\setup.ps1 to install dependencies"
        return $false
    }
    
    # Check for other critical dependencies
    $missingDeps = @()
    $deps = @("accelerate", "diffusers", "transformers")
    foreach ($dep in $deps) {
        try {
            & $pythonCmd -c "import $dep" 2>&1 | Out-Null
            if ($LASTEXITCODE -ne 0) {
                $missingDeps += $dep
            }
        } catch {
            $missingDeps += $dep
        }
    }
    
    if ($missingDeps.Count -gt 0) {
        Write-Warning "Missing dependencies: $($missingDeps -join ', ')"
        Write-Info "Install with: pip install -r requirements.txt"
        return $false
    } else {
        Write-Success "Core dependencies verified"
        return $true
    }
}

# Parse arguments
function Parse-Args {
    if ($args.Count -eq 0) {
        Show-Usage
        exit 0
    }
    
    $script:MODE = $args[0]
    $remainingArgs = $args[1..($args.Count - 1)]
    
    $i = 0
    while ($i -lt $remainingArgs.Count) {
        $arg = $remainingArgs[$i]
        switch ($arg) {
            { $_ -in "-r", "--recover" } {
                $script:RECOVER = $true
            }
            { $_ -in "-n", "--name" } {
                $script:JOB_NAME = $remainingArgs[$i + 1]
                $i++
            }
            { $_ -in "-l", "--log" } {
                $script:LOG_FILE = $remainingArgs[$i + 1]
                $i++
            }
            { $_ -in "-p", "--port" } {
                $script:UI_PORT = $remainingArgs[$i + 1]
                $i++
            }
            "--dev" {
                if ($MODE -eq "ui") {
                    $script:UI_DEV_MODE = $true
                } else {
                    Write-Warning "--dev flag is only valid for 'ui' mode"
                }
            }
            default {
                if ($MODE -eq "train") {
                    if ([string]::IsNullOrEmpty($CONFIG_FILE)) {
                        $script:CONFIG_FILE = $arg
                    } else {
                        $script:CONFIG_FILE = "$CONFIG_FILE $arg"
                    }
                }
            }
        }
        $i++
    }
    
    # Validate config file for train mode
    if ($MODE -eq "train" -and [string]::IsNullOrEmpty($CONFIG_FILE)) {
        Write-Error "No config file specified for training mode"
        Show-Usage
        exit 1
    }
}

# Main execution
function Main {
    Write-Info "AI Toolkit Startup Script"
    Write-Info "=========================="
    Write-Host ""
    
    # Parse arguments
    Parse-Args $args
    
    # Setup environment
    Setup-Venv
    
    # For setup mode, skip dependency verification initially
    if ($MODE -ne "setup") {
        Detect-Backend
        if (-not (Verify-Dependencies)) {
            Write-Error "Dependencies not satisfied. Run '.\start_toolkit.ps1 setup' to install them."
            exit 1
        }
    }
    
    Write-Host ""
    Write-Info "Starting in mode: $MODE"
    Write-Host ""
    
    # Execute based on mode
    switch ($MODE) {
        "setup" {
            Write-Info "Setting up/validating toolkit environment..."
            Write-Info "Please run .\setup.ps1 for full setup"
            & .\setup.ps1
        }
        
        "train" {
            Write-Info "Running training job(s)..."
            $pythonCmd = Get-PythonCmd
            $cmdArgs = @("run.py")
            
            # Add config files
            foreach ($config in $CONFIG_FILE.Split(' ')) {
                if (-not [string]::IsNullOrEmpty($config)) {
                    $cmdArgs += $config
                }
            }
            
            # Add options
            if ($RECOVER) {
                $cmdArgs += "--recover"
            }
            
            if (-not [string]::IsNullOrEmpty($JOB_NAME)) {
                $cmdArgs += "--name"
                $cmdArgs += $JOB_NAME
            }
            
            if (-not [string]::IsNullOrEmpty($LOG_FILE)) {
                $cmdArgs += "--log"
                $cmdArgs += $LOG_FILE
            }
            
            Write-Info "Command: $pythonCmd $($cmdArgs -join ' ')"
            & $pythonCmd $cmdArgs
        }
        
        "gradio" {
            Write-Info "Launching Gradio UI..."
            $pythonCmd = Get-PythonCmd
            try {
                & $pythonCmd -c "import gradio" 2>&1 | Out-Null
                if ($LASTEXITCODE -ne 0) {
                    Write-Error "Gradio is not installed!"
                    Write-Info "Install with: pip install gradio"
                    exit 1
                }
            } catch {
                Write-Error "Gradio is not installed!"
                Write-Info "Install with: pip install gradio"
                exit 1
            }
            & $pythonCmd flux_train_ui.py
        }
        
        "ui" {
            Write-Info "Launching web UI on port $UI_PORT..."
            if (-not (Test-Path "ui")) {
                Write-Error "UI directory not found!"
                exit 1
            }
            
            Push-Location ui
            if (-not (Test-Path "node_modules")) {
                Write-Info "Installing UI dependencies..."
                npm install
            }
            
            # Check if --dev flag is set for development mode with hot reload
            if ($UI_DEV_MODE) {
                Write-Info "Starting UI in DEVELOPMENT mode (hot reload enabled)..."
                Write-Info "UI will be available at http://localhost:3000 (or next available port)"
                $env:PORT = $UI_PORT
                npm run dev
            } else {
                Write-Info "Starting UI in PRODUCTION mode..."
                Write-Info "To use dev mode with hot reload, run: .\start_toolkit.ps1 ui --dev"
                $env:PORT = $UI_PORT
                npm run build_and_start
            }
            Pop-Location
        }
        
        "help" {
            Show-Usage
        }
        
        default {
            Write-Error "Unknown mode: $MODE"
            Show-Usage
            exit 1
        }
    }
}

# Run main function
Main $args




