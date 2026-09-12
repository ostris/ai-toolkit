#!/bin/bash
# Setup script for AI Toolkit on Linux
# This script sets up the environment, installs dependencies, and validates the installation

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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

print_info "AI Toolkit Setup Script"
print_info "======================"
echo ""

# Check if we're in the right directory
if [ ! -f "run.py" ] || [ ! -f "requirements.txt" ]; then
    print_error "Please run this script from the ai-toolkit root directory"
    exit 1
fi

# Check for Python
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed. Please install Python 3.10 or higher."
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 10 ]); then
    print_error "Python 3.10 or higher is required. Found: $PYTHON_VERSION"
    exit 1
fi

print_success "Python $PYTHON_VERSION detected"

# Create virtual environment if it doesn't exist
if [ -d ".venv" ] || [ -d "venv" ]; then
    print_info "Virtual environment already exists"
    if [ -d ".venv" ]; then
        source .venv/bin/activate
        PYTHON_CMD=".venv/bin/python"
    else
        source venv/bin/activate
        PYTHON_CMD="venv/bin/python"
    fi
else
    print_info "Creating virtual environment..."
    if command -v uv &> /dev/null; then
        print_info "Using uv to create virtual environment..."
        uv venv
        source .venv/bin/activate
        PYTHON_CMD=".venv/bin/python"
    else
        print_info "Using python3 -m venv to create virtual environment..."
        python3 -m venv venv
        source venv/bin/activate
        PYTHON_CMD="venv/bin/python"
    fi
    print_success "Virtual environment created"
fi

# Upgrade pip
print_info "Upgrading pip..."
if command -v uv &> /dev/null; then
    uv pip install --upgrade pip
else
    $PYTHON_CMD -m pip install --upgrade pip
fi

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

# Detect GPU backend
print_info "Detecting GPU backend..."
if command -v rocm-smi &> /dev/null; then
    print_info "ROCm detected. Installing PyTorch with ROCm support..."
    
    # Get GPU architecture
    if [ -z "$PYTORCH_ROCM_ARCH" ]; then
        print_info "Detecting GPU architecture..."
        GPU_INFO=$(rocm-smi --showproductname 2>/dev/null || echo "")
        
        # Extract gfx architecture code from rocm-smi output
        # rocm-smi output format varies, try to extract gfx#### pattern
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
            read -p "Enter your GPU architecture [gfx1151]: " ROCM_ARCH
            ROCM_ARCH=${ROCM_ARCH:-gfx1151}
        fi
        export PYTORCH_ROCM_ARCH=$ROCM_ARCH
        print_info "Using GPU architecture: $ROCM_ARCH"
    else
        ROCM_ARCH=$PYTORCH_ROCM_ARCH
        print_info "Using GPU architecture from environment: $ROCM_ARCH"
    fi
    
    # Install PyTorch with ROCm
    print_info "Installing PyTorch with ROCm support..."
    if command -v uv &> /dev/null; then
        uv pip install --upgrade --index-url "https://rocm.nightlies.amd.com/v2/${ROCM_ARCH}/" --pre torch torchaudio torchvision
    else
        pip install --upgrade --index-url "https://rocm.nightlies.amd.com/v2/${ROCM_ARCH}/" --pre torch torchaudio torchvision
    fi
else
    print_info "CUDA/NVIDIA detected or no GPU. Installing PyTorch with CUDA support..."
    if command -v uv &> /dev/null; then
        uv pip install --no-cache-dir torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu126
    else
        pip install --no-cache-dir torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu126
    fi
fi

# Install other dependencies
print_info "Installing other dependencies from requirements.txt..."
if command -v uv &> /dev/null; then
    uv pip install -r requirements.txt
else
    pip install -r requirements.txt
fi

# Verify installation
print_info "Verifying installation..."
if $PYTHON_CMD -c "import torch" 2>/dev/null; then
    print_success "PyTorch installed successfully"
    
    # Check GPU availability
    if $PYTHON_CMD -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
        DEVICE_NAME=$($PYTHON_CMD -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null)
        print_success "GPU detected: $DEVICE_NAME"
    else
        print_warning "No GPU detected. Training will run on CPU (very slow)."
    fi
else
    print_error "PyTorch installation failed"
    exit 1
fi

# Check other critical dependencies
MISSING_DEPS=()
for dep in "accelerate" "diffusers" "transformers"; do
    if ! $PYTHON_CMD -c "import $dep" 2>/dev/null; then
        MISSING_DEPS+=("$dep")
    fi
done

if [ ${#MISSING_DEPS[@]} -gt 0 ]; then
    print_warning "Some dependencies are missing: ${MISSING_DEPS[*]}"
    print_info "Try running: pip install -r requirements.txt"
else
    print_success "All core dependencies verified"
fi

print_success "Setup complete!"
print_info "You can now run the toolkit with: ./start_toolkit.sh ui"
print_info "Or start training with: ./start_toolkit.sh train config/your_config.yaml"

