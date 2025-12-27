#!/bin/bash
# Startup script for AI Toolkit with ROCm/CUDA support
# This script sets up the environment and launches the toolkit

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
MODE="help"
CONFIG_FILE=""
RECOVER=false
JOB_NAME=""
LOG_FILE=""
UI_PORT=8675

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to map GPU architecture to ROCm directory name
# ROCm nightlies use specific directory names that may differ from the detected architecture
map_gpu_arch_to_rocm_dir() {
    local detected_arch="$1"
    
    # Map gfx110X variants to gfx110X-all (RDNA 3 architecture)
    # This includes gfx1100, gfx1101, gfx1102, gfx1103
    if echo "$detected_arch" | grep -qE "gfx110[0-3]"; then
        echo "gfx110X-all"
        return
    fi
    
    # Direct mappings for architectures that match their directory names
    # gfx1151 (RDNA 3.5), gfx1030 (RDNA 2), gfx90a (CDNA 2), gfx906 (Vega), etc.
    case "$detected_arch" in
        gfx1151|gfx1030|gfx90a|gfx906|gfx908|gfx941|gfx942)
            echo "$detected_arch"
            return
            ;;
        *)
            # For unknown architectures, try the detected name first
            # User can override if it doesn't work
            echo "$detected_arch"
            return
            ;;
    esac
}

# Function to show usage
show_usage() {
    echo "AI Toolkit Startup Script"
    echo "=========================="
    echo ""
    echo "Usage: $0 [MODE] [OPTIONS]"
    echo ""
    echo "Modes:"
    echo "  setup                                    - Setup/validate the toolkit environment"
    echo "  train <config_file> [config_file2 ...]  - Run training job(s) with config file(s)"
    echo "  gradio                                  - Launch Gradio UI for FLUX training"
    echo "  ui                                       - Launch web UI (Next.js, production mode)"
    echo "  ui --dev                                 - Launch web UI (development mode with hot reload)"
    echo "  help                                     - Show this help message"
    echo ""
    echo "Training Options:"
    echo "  -r, --recover                            - Continue running additional jobs if one fails"
    echo "  -n, --name NAME                          - Name to replace [name] tag in config"
    echo "  -l, --log FILE                           - Log file to write output to"
    echo ""
    echo "UI Options:"
    echo "  -p, --port PORT                          - Port for web UI (default: 8675)"
    echo ""
    echo "Examples:"
    echo "  $0 train config/examples/train_lora_wan22_14b_24gb.yaml"
    echo "  $0 train config/my_config.yaml -r -n my_training_run"
    echo "  $0 gradio"
    echo "  $0 ui"
    echo ""
}

# Function to detect backend and set ROCm environment variables
detect_backend() {
    print_info "Detecting GPU backend..."
    
    # Ensure PYTHON_CMD is set
    if [ -z "$PYTHON_CMD" ]; then
        PYTHON_CMD=$(get_python)
        export PYTHON_CMD
    fi
    
    BACKEND_OUTPUT=$($PYTHON_CMD -c "import torch; print('ROCm' if hasattr(torch.version, 'hip') and torch.version.hip else 'CUDA')" 2>&1 | grep -E '^(ROCm|CUDA)$' | head -1)
    if [ -n "$BACKEND_OUTPUT" ]; then
        BACKEND="$BACKEND_OUTPUT"
        print_success "Backend detected: $BACKEND"
        
        if [ "$BACKEND" = "ROCm" ]; then
            # Set ROCm environment variables if not already set
            if [ -z "$ROCM_PATH" ]; then
                if [ -d "/opt/rocm" ]; then
                    export ROCM_PATH=/opt/rocm
                else
                    print_warning "ROCM_PATH not set and /opt/rocm not found. ROCm may not work correctly."
                fi
            fi
            
            if [ -z "$PYTORCH_ROCM_ARCH" ]; then
                export PYTORCH_ROCM_ARCH="gfx1151"
                print_info "Set PYTORCH_ROCM_ARCH=gfx1151"
            fi
            
            # ROCBLAS_USE_HIPBLASLT can cause HIPBLAS_STATUS_INTERNAL_ERROR with quantized models
            # Disable by default - can be enabled via environment variable if needed
            if [ -z "$ROCBLAS_USE_HIPBLASLT" ]; then
                export ROCBLAS_USE_HIPBLASLT=0
                print_info "Set ROCBLAS_USE_HIPBLASLT=0 (disabled to avoid quantized model crashes)"
            fi
            
            # Reduce ROCBLAS logging overhead
            if [ -z "$ROCBLAS_LOG_LEVEL" ]; then
                export ROCBLAS_LOG_LEVEL=0
                print_info "Set ROCBLAS_LOG_LEVEL=0 (disable verbose logging)"
            fi
            
            # Set AMD_SERIALIZE_KERNEL for better error reporting (as suggested by HIP errors)
            if [ -z "$AMD_SERIALIZE_KERNEL" ]; then
                export AMD_SERIALIZE_KERNEL=3
                print_info "Set AMD_SERIALIZE_KERNEL=3 for better error reporting"
            fi
            
            # Set TORCH_USE_HIP_DSA to enable device-side assertions (as suggested by HIP errors)
            if [ -z "$TORCH_USE_HIP_DSA" ]; then
                export TORCH_USE_HIP_DSA=1
                print_info "Set TORCH_USE_HIP_DSA=1 for device-side assertions"
            fi
            
            # Set HSA_OVERRIDE_GFX_VERSION for Strix Halo (gfx1151) compatibility
            if [ -z "$HSA_OVERRIDE_GFX_VERSION" ]; then
                export HSA_OVERRIDE_GFX_VERSION=11.0.0
                print_info "Set HSA_OVERRIDE_GFX_VERSION=11.0.0 for gfx1151 compatibility"
            fi
            
            # Set HIP_LAUNCH_BLOCKING for debugging (optional, can be disabled for performance)
            # Setting to 1 makes kernel launches synchronous for better error reporting
            if [ -z "$HIP_LAUNCH_BLOCKING" ]; then
                # Default to 0 for performance, but can be set to 1 for debugging
                export HIP_LAUNCH_BLOCKING="${HIP_LAUNCH_BLOCKING:-0}"
                if [ "$HIP_LAUNCH_BLOCKING" = "1" ]; then
                    print_info "HIP_LAUNCH_BLOCKING=1 (synchronous kernels for debugging)"
                fi
            fi
            
            # Additional ROCm tuning for APU/quantization (from Grok's analysis)
            # Disable SDMA if conflicting on APUs (common for memcpy in quant)
            if [ -z "$HSA_ENABLE_SDMA" ]; then
                export HSA_ENABLE_SDMA=0
                print_info "Set HSA_ENABLE_SDMA=0 (disable SDMA for APU compatibility)"
            fi
            
            # Better VRAM fragmentation for large shared memory pools (128GB EVO-X2)
            if [ -z "$PYTORCH_ROCM_ALLOC_CONF" ]; then
                export PYTORCH_ROCM_ALLOC_CONF="max_split_size_mb:768,garbage_collect=1"
                print_info "Set PYTORCH_ROCM_ALLOC_CONF for better VRAM fragmentation"
            fi
            
            # Additional ROCm optimization variables
            # HIP_VISIBLE_DEVICES can be used to select specific GPUs (similar to CUDA_VISIBLE_DEVICES)
            if [ -n "$HIP_VISIBLE_DEVICES" ]; then
                print_info "HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES"
            fi
            
            # Set library paths
            if [ -n "$ROCM_PATH" ]; then
                export LD_LIBRARY_PATH=$ROCM_PATH/lib:${LD_LIBRARY_PATH:-}
                export DEVICE_LIB_PATH=$ROCM_PATH/llvm/amdgcn/bitcode
                export HIP_DEVICE_LIB_PATH=$ROCM_PATH/llvm/amdgcn/bitcode
                
                # Add ROCm binaries to PATH if not already there
                if [[ ":$PATH:" != *":$ROCM_PATH/bin:"* ]]; then
                    export PATH=$ROCM_PATH/bin:$PATH
                fi
            fi
            
            # Print configuration summary
            print_info "ROCm Configuration Summary:"
            print_info "  - PYTORCH_ROCM_ARCH: ${PYTORCH_ROCM_ARCH}"
            print_info "  - ROCBLAS_USE_HIPBLASLT: ${ROCBLAS_USE_HIPBLASLT}"
            print_info "  - HSA_OVERRIDE_GFX_VERSION: ${HSA_OVERRIDE_GFX_VERSION}"
            print_info "  - AMD_SERIALIZE_KERNEL: ${AMD_SERIALIZE_KERNEL}"
            print_info "  - TORCH_USE_HIP_DSA: ${TORCH_USE_HIP_DSA}"
            print_info "  - HIP_LAUNCH_BLOCKING: ${HIP_LAUNCH_BLOCKING}"
            print_info "  - HSA_ENABLE_SDMA: ${HSA_ENABLE_SDMA}"
            print_info "  - PYTORCH_ROCM_ALLOC_CONF: ${PYTORCH_ROCM_ALLOC_CONF}"
        fi
        
        # Verify GPU is available
        if $PYTHON_CMD -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
            DEVICE_NAME=$($PYTHON_CMD -c "import torch; print(torch.cuda.get_device_name(0))" 2>&1 | grep -v "failed to run amdgpu-arch" | head -1)
            print_success "GPU detected: $DEVICE_NAME"
        else
            print_warning "No GPU detected. Training will run on CPU (very slow)."
        fi
    else
        print_warning "PyTorch not available. Make sure it's installed."
        BACKEND="UNKNOWN"
    fi
}

# Global variable to store Python command
PYTHON_CMD=""

# Function to get Python executable path
get_python() {
    # If we're in a virtual environment, use its Python
    if [ -n "$VIRTUAL_ENV" ]; then
        if [ -f "$VIRTUAL_ENV/bin/python" ]; then
            echo "$VIRTUAL_ENV/bin/python"
            return 0
        fi
    fi
    
    # Check for uv venv
    if [ -d ".venv" ] && [ -f ".venv/bin/python" ]; then
        echo ".venv/bin/python"
        return 0
    fi
    
    # Check for standard venv
    if [ -d "venv" ] && [ -f "venv/bin/python" ]; then
        echo "venv/bin/python"
        return 0
    fi
    
    # Fall back to system python3
    echo "python3"
}

# Function to check and activate virtual environment
setup_venv() {
    # Check if we're already in a virtual environment
    if [ -n "$VIRTUAL_ENV" ]; then
        print_info "Virtual environment already active: $VIRTUAL_ENV"
        PYTHON_CMD=$(get_python)
        export PYTHON_CMD
        return 0
    fi
    
    # Check for uv venv
    if [ -d ".venv" ]; then
        print_info "Activating uv virtual environment..."
        source .venv/bin/activate
        print_success "Virtual environment activated"
        PYTHON_CMD=$(get_python)
        export PYTHON_CMD
        return 0
    fi
    
    # Check for standard venv
    if [ -d "venv" ]; then
        print_info "Activating virtual environment..."
        source venv/bin/activate
        print_success "Virtual environment activated"
        PYTHON_CMD=$(get_python)
        export PYTHON_CMD
        return 0
    fi
    
    print_warning "No virtual environment found. Using system Python."
    print_info "Consider creating one with: uv venv or python -m venv venv"
    PYTHON_CMD=$(get_python)
    export PYTHON_CMD
}

# Function to verify dependencies
verify_dependencies() {
    print_info "Verifying dependencies..."
    
    # Ensure PYTHON_CMD is set
    if [ -z "$PYTHON_CMD" ]; then
        PYTHON_CMD=$(get_python)
        export PYTHON_CMD
    fi
    
    if ! $PYTHON_CMD -c "import torch" 2>/dev/null; then
        print_error "PyTorch is not installed!"
        print_info "Run: ./setup.sh to install dependencies"
        print_info "Or manually: pip install torch torchvision torchaudio"
        return 1
    fi
    
    # Check for other critical dependencies
    MISSING_DEPS=()
    for dep in "accelerate" "diffusers" "transformers"; do
        if ! $PYTHON_CMD -c "import $dep" 2>/dev/null; then
            MISSING_DEPS+=("$dep")
        fi
    done
    
    if [ ${#MISSING_DEPS[@]} -gt 0 ]; then
        print_warning "Missing dependencies: ${MISSING_DEPS[*]}"
        print_info "Install with: pip install -r requirements.txt"
        return 1
    else
        print_success "Core dependencies verified"
        return 0
    fi
}

# Parse arguments
parse_args() {
    if [ $# -eq 0 ]; then
        show_usage
        exit 0
    fi
    
    MODE=$1
    shift
    
    case "$MODE" in
        setup|train)
            if [ "$MODE" = "train" ] && [ $# -eq 0 ]; then
                print_error "No config file specified for training mode"
                show_usage
                exit 1
            fi
            # Don't set CONFIG_FILE here - collect it after parsing options
            ;;
        gradio|ui|help)
            # No additional args needed for these modes
            ;;
        *)
            print_error "Unknown mode: $MODE"
            show_usage
            exit 1
            ;;
    esac
    
    # Initialize UI_DEV_MODE flag
    UI_DEV_MODE=false
    
    # Parse remaining options and collect config files
    while [[ $# -gt 0 ]]; do
        case $1 in
            -r|--recover)
                RECOVER=true
                shift
                ;;
            -n|--name)
                JOB_NAME="$2"
                shift 2
                ;;
            -l|--log)
                LOG_FILE="$2"
                shift 2
                ;;
            -p|--port)
                UI_PORT="$2"
                shift 2
                ;;
            --dev)
                if [ "$MODE" = "ui" ]; then
                    UI_DEV_MODE=true
                else
                    print_warning "--dev flag is only valid for 'ui' mode"
                fi
                shift
                ;;
            *)
                # For train mode, remaining args are config files
                if [ "$MODE" = "train" ]; then
                    if [ -z "$CONFIG_FILE" ]; then
                        CONFIG_FILE="$1"
                    else
                        CONFIG_FILE="$CONFIG_FILE $1"
                    fi
                fi
                shift
                ;;
        esac
    done
    
    # Validate config file for train mode
    if [ "$MODE" = "train" ] && [ -z "$CONFIG_FILE" ]; then
        print_error "No config file specified for training mode"
        show_usage
        exit 1
    fi
}

# Main execution
main() {
    print_info "AI Toolkit Startup Script"
    print_info "=========================="
    echo ""
    
    # Parse arguments
    parse_args "$@"
    
    # Setup environment
    setup_venv
    
    # For setup and help modes, skip dependency verification
    # For UI mode, skip detect_backend - let run.py set ROCm vars when jobs are spawned
    if [ "$MODE" != "setup" ] && [ "$MODE" != "help" ] && [ "$MODE" != "ui" ]; then
        detect_backend
        if ! verify_dependencies; then
            print_error "Dependencies not satisfied. Run './start_toolkit.sh setup' to install them."
            exit 1
        fi
    elif [ "$MODE" = "ui" ]; then
        # For UI mode, only verify dependencies but don't set ROCm vars
        # ROCm vars will be set by run.py when jobs are spawned
        if ! verify_dependencies; then
            print_error "Dependencies not satisfied. Run './start_toolkit.sh setup' to install them."
            exit 1
        fi
    fi
    
    echo ""
    print_info "Starting in mode: $MODE"
    echo ""
    
    # Execute based on mode
    case "$MODE" in
        setup)
            print_info "Setting up/validating toolkit environment..."
            
            # Ensure PYTHON_CMD is set
            if [ -z "$PYTHON_CMD" ]; then
                PYTHON_CMD=$(get_python)
                export PYTHON_CMD
            fi
            
            # Check if we need to create venv
            if [ ! -d ".venv" ] && [ ! -d "venv" ] && [ -z "$VIRTUAL_ENV" ]; then
                print_info "No virtual environment found. Creating one..."
                if command -v uv &> /dev/null; then
                    print_info "Using uv to create virtual environment..."
                    uv venv
                    source .venv/bin/activate
                    PYTHON_CMD=$(get_python)
                    export PYTHON_CMD
                else
                    print_info "Using python3 -m venv to create virtual environment..."
                    python3 -m venv venv
                    source venv/bin/activate
                    PYTHON_CMD=$(get_python)
                    export PYTHON_CMD
                fi
            fi
            
            # Ensure PYTHON_CMD is set
            if [ -z "$PYTHON_CMD" ]; then
                PYTHON_CMD=$(get_python)
                export PYTHON_CMD
            fi
            
            # Check PyTorch
            print_info "Checking PyTorch installation..."
            if ! $PYTHON_CMD -c "import torch" 2>/dev/null; then
                print_warning "PyTorch not installed. Detecting GPU backend..."
                
                # Try to detect if ROCm or CUDA
                if command -v rocm-smi &> /dev/null; then
                    print_info "ROCm detected. Installing PyTorch with ROCm support..."
                    
                    # Try to auto-detect GPU architecture
                    print_info "Detecting GPU architecture..."
                    GPU_INFO=$(rocm-smi --showproductname 2>/dev/null || echo "")
                    
                    # Extract gfx architecture code from rocm-smi output
                    DETECTED_ARCH=""
                    if echo "$GPU_INFO" | grep -qE "gfx[0-9]+"; then
                        # Extract the first gfx#### pattern found
                        DETECTED_ARCH=$(echo "$GPU_INFO" | grep -oE "gfx[0-9]+" | head -1)
                    fi
                    
                    if [ -n "$DETECTED_ARCH" ]; then
                        ROCM_ARCH=$(map_gpu_arch_to_rocm_dir "$DETECTED_ARCH")
                        print_info "Detected GPU architecture: $DETECTED_ARCH"
                        print_info "Mapped to ROCm directory: $ROCM_ARCH"
                    else
                        print_warning "Could not auto-detect GPU architecture from rocm-smi output"
                        print_info "Common architectures:"
                        print_info "  - gfx1151 (RDNA 3.5 - Strix Point Halo APU)"
                        print_info "  - gfx110X-all (RDNA 3 - RX 7900/7800/7700 series, gfx1100/gfx1101/gfx1102/gfx1103)"
                        print_info "  - gfx1030 (RDNA 2 - RX 6900/6800/6700 series)"
                        print_info "  - gfx90a (CDNA 2 - Instinct MI200 series)"
                        read -p "Enter GPU architecture [gfx1151]: " ROCM_ARCH
                        ROCM_ARCH=${ROCM_ARCH:-gfx1151}
                    fi
                    
                    if command -v uv &> /dev/null; then
                        uv pip install --upgrade --index-url "https://rocm.nightlies.amd.com/v2/${ROCM_ARCH}/" --pre torch torchaudio torchvision
                    else
                        pip install --upgrade --index-url "https://rocm.nightlies.amd.com/v2/${ROCM_ARCH}/" --pre torch torchaudio torchvision
                    fi
                else
                    print_info "Installing PyTorch with CUDA support..."
                    if command -v uv &> /dev/null; then
                        uv pip install --no-cache-dir torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu126
                    else
                        pip install --no-cache-dir torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu126
                    fi
                fi
            else
                print_success "PyTorch is installed"
            fi
            
            # Install other dependencies
            print_info "Installing/updating other dependencies..."
            if command -v uv &> /dev/null; then
                uv pip install -r requirements.txt
            else
                pip install -r requirements.txt
            fi
            
            # Verify everything
            print_info "Verifying installation..."
            detect_backend
            if verify_dependencies; then
                print_success "Setup complete! Toolkit is ready to use."
            else
                print_error "Setup completed but some dependencies are missing."
                exit 1
            fi
            ;;
            
        train)
            print_info "Running training job(s)..."
            # Ensure PYTHON_CMD is set
            if [ -z "$PYTHON_CMD" ]; then
                PYTHON_CMD=$(get_python)
                export PYTHON_CMD
            fi
            CMD="$PYTHON_CMD run.py"
            
            # Add config files
            for config in $CONFIG_FILE; do
                CMD="$CMD \"$config\""
            done
            
            # Add options
            if [ "$RECOVER" = true ]; then
                CMD="$CMD --recover"
            fi
            
            if [ -n "$JOB_NAME" ]; then
                CMD="$CMD --name \"$JOB_NAME\""
            fi
            
            if [ -n "$LOG_FILE" ]; then
                CMD="$CMD --log \"$LOG_FILE\""
            fi
            
            print_info "Command: $CMD"
            eval $CMD
            ;;
            
        gradio)
            print_info "Launching Gradio UI..."
            # Ensure PYTHON_CMD is set
            if [ -z "$PYTHON_CMD" ]; then
                PYTHON_CMD=$(get_python)
                export PYTHON_CMD
            fi
            if ! $PYTHON_CMD -c "import gradio" 2>/dev/null; then
                print_error "Gradio is not installed!"
                print_info "Install with: pip install gradio"
                exit 1
            fi
            $PYTHON_CMD flux_train_ui.py
            ;;
            
        ui)
            print_info "Launching web UI on port $UI_PORT..."
            if [ ! -d "ui" ]; then
                print_error "UI directory not found!"
                exit 1
            fi
            
            # For UI mode, do NOT set ROCm environment variables
            # They will be set by run.py when jobs are spawned
            # Unset any ROCm vars that might have been set previously to avoid conflicts
            unset AMD_SERIALIZE_KERNEL
            unset TORCH_USE_HIP_DSA
            unset HSA_ENABLE_SDMA
            unset PYTORCH_ROCM_ALLOC_CONF
            unset ROCBLAS_USE_HIPBLASLT
            unset ROCBLAS_LOG_LEVEL
            unset HSA_OVERRIDE_GFX_VERSION
            # Note: We keep PYTORCH_ROCM_ARCH and HIP_LAUNCH_BLOCKING as they might be needed
            # but run.py will override them if needed
            
            cd ui
            if [ ! -d "node_modules" ]; then
                print_info "Installing UI dependencies..."
                npm install
            fi
            
            # Check if --dev flag is set for development mode with hot reload
            if [ "$UI_DEV_MODE" = "true" ]; then
                print_info "Starting UI in DEVELOPMENT mode (hot reload enabled)..."
                print_info "UI will be available at http://localhost:3000 (or next available port)"
                PORT=$UI_PORT npm run dev
            else
                print_info "Starting UI in PRODUCTION mode..."
                print_info "To use dev mode with hot reload, run: ./start_toolkit.sh ui --dev"
                PORT=$UI_PORT npm run build_and_start
            fi
            ;;
            
        help)
            show_usage
            ;;
    esac
}

# Run main function
main "$@"

