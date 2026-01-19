{
  description = "AI Toolkit by Ostris - Training toolkit for diffusion models";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config = {
            allowUnfree = true;
            cudaSupport = true;
          };
        };

        python = pkgs.python312;

        # Option 1: Use PyTorch nightly for Blackwell (sm_120) support
        setupVenvNightly = pkgs.writeShellScriptBin "setup-venv-nightly" ''
          set -e
          
          if [ ! -d "venv" ]; then
            echo "Creating Python virtual environment..."
            ${python}/bin/python -m venv venv
            
            echo "Activating virtual environment..."
            source venv/bin/activate
            
            echo "Upgrading pip..."
            pip install --upgrade pip setuptools wheel
            
            echo "Installing PyTorch Nightly with CUDA 12.8 support (for Blackwell sm_120)..."
            pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
            
            echo "Installing requirements from requirements.txt..."
            if [ -f "requirements.txt" ]; then
              pip install -r requirements.txt
            else
              echo "Warning: requirements.txt not found. Installing common dependencies..."
              pip install accelerate transformers diffusers huggingface_hub
              pip install safetensors omegaconf pillow tqdm pyyaml
              pip install bitsandbytes triton xformers
            fi
            
            echo ""
            echo "✅ Virtual environment setup complete with PyTorch Nightly!"
            echo "⚠️  Note: Nightly builds are experimental and may have bugs."
            echo "To activate, run: source venv/bin/activate"
          else
            echo "Virtual environment already exists."
            echo "To recreate, delete the 'venv' directory first."
          fi
        '';

        # Option 2: Use PyTorch stable (2.6.0) - won't work with Blackwell but stable
        setupVenvStable = pkgs.writeShellScriptBin "setup-venv-stable" ''
          set -e
          
          if [ ! -d "venv" ]; then
            echo "Creating Python virtual environment..."
            ${python}/bin/python -m venv venv
            
            echo "Activating virtual environment..."
            source venv/bin/activate
            
            echo "Upgrading pip..."
            pip install --upgrade pip setuptools wheel
            
            echo "Installing PyTorch 2.6.0 (stable) with CUDA 12.4..."
            echo "⚠️  WARNING: PyTorch 2.6.0 does NOT support Blackwell (sm_120)."
            echo "⚠️  Your RTX PRO 6000 will NOT work with this version!"
            echo "⚠️  Use 'setup-venv-nightly' instead for Blackwell support."
            echo ""
            read -p "Press Enter to continue anyway, or Ctrl+C to cancel..."
            
            pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
            
            echo "Installing requirements from requirements.txt..."
            if [ -f "requirements.txt" ]; then
              pip install -r requirements.txt
            else
              echo "Warning: requirements.txt not found. Installing common dependencies..."
              pip install accelerate transformers diffusers huggingface_hub
              pip install safetensors omegaconf pillow tqdm pyyaml
              pip install bitsandbytes triton xformers
            fi
            
            echo ""
            echo "✅ Virtual environment setup complete with PyTorch 2.6.0 (stable)."
            echo "⚠️  This version does NOT support your Blackwell GPU!"
            echo "To activate, run: source venv/bin/activate"
          else
            echo "Virtual environment already exists."
            echo "To recreate, delete the 'venv' directory first."
          fi
        '';

        activateVenv = pkgs.writeShellScriptBin "activate-venv" ''
          if [ -f "venv/bin/activate" ]; then
            source venv/bin/activate
            echo "Virtual environment activated."
          else
            echo "Error: Virtual environment not found."
            echo "Run 'setup-venv-nightly' or 'setup-venv-stable' first."
            exit 1
          fi
        '';

        runTraining = pkgs.writeShellScriptBin "run-training" ''
          if [ -f "venv/bin/activate" ]; then
            source venv/bin/activate
            
            # Set up library paths for OpenCV and CUDA
            export LD_LIBRARY_PATH="${pkgs.stdenv.cc.cc.lib}/lib:${pkgs.cudaPackages.cudatoolkit}/lib:${pkgs.cudaPackages.cudnn}/lib:${pkgs.linuxPackages.nvidia_x11}/lib:${pkgs.libGL}/lib:${pkgs.glib}/lib:${pkgs.glibc}/lib:${pkgs.xorg.libX11}/lib:${pkgs.xorg.libXext}/lib:$LD_LIBRARY_PATH"
            
            python run.py "$@"
          else
            echo "Error: Virtual environment not found."
            echo "Run 'setup-venv-nightly' or 'setup-venv-stable' first."
            exit 1
          fi
        '';

        startUI = pkgs.writeShellScriptBin "start-ui" ''
          if [ ! -d "ui/node_modules" ]; then
            echo "Installing UI dependencies..."
            cd ui
            ${pkgs.nodejs}/bin/npm install
            cd ..
          fi
          
          if [ -f "venv/bin/activate" ]; then
            source venv/bin/activate
            
            # Set up library paths
            export LD_LIBRARY_PATH="${pkgs.stdenv.cc.cc.lib}/lib:${pkgs.libGL}/lib:${pkgs.glib}/lib:${pkgs.glibc}/lib:${pkgs.xorg.libX11}/lib:${pkgs.xorg.libXext}/lib:$LD_LIBRARY_PATH"
            
            cd ui
            ${pkgs.nodejs}/bin/npm run build_and_start
          else
            echo "Error: Virtual environment not found."
            echo "Run 'setup-venv-nightly' or 'setup-venv-stable' first."
            exit 1
          fi
        '';

        systemDeps = with pkgs; [
          python

          cudaPackages.cudatoolkit
          cudaPackages.cudnn
          linuxPackages.nvidia_x11

          gcc
          gnumake
          cmake
          pkg-config

          git
          git-lfs

          nodejs

          which
          procps

          # Graphics and GUI libraries for OpenCV
          libGL
          libGLU
          glib
          glibc
          
          # Image processing libraries
          zlib
          libjpeg
          libpng
          libtiff
          
          # Additional libraries for OpenCV
          xorg.libX11
          xorg.libXext
          libv4l
          ffmpeg

          mkl
        ];

      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = systemDeps ++ [
            setupVenvNightly
            setupVenvStable
            activateVenv
            runTraining
            startUI
            pkgs.prisma-engines_6
            pkgs.prisma_6
          ];

          shellHook = ''
            echo "🚀 AI Toolkit Development Environment"
            echo "======================================"
            echo ""
            echo "Available commands:"
            echo "  setup-venv-nightly  - Setup with PyTorch Nightly (RECOMMENDED for Blackwell)"
            echo "  setup-venv-stable   - Setup with PyTorch 2.6.0 (NOT compatible with Blackwell)"
            echo "  activate-venv       - Activate the virtual environment"
            echo "  run-training        - Run training with config file"
            echo "  start-ui            - Start the web UI"
            echo ""
            echo "Quick start for Blackwell GPU:"
            echo "  1. Clone the repository if you haven't:"
            echo "     git clone https://github.com/ostris/ai-toolkit.git"
            echo "     cd ai-toolkit"
            echo ""
            echo "  2. Initialize submodules:"
            echo "     git submodule update --init --recursive"
            echo ""
            echo "  3. Setup Python environment with Nightly:"
            echo "     setup-venv-nightly"
            echo ""
            echo "  4. Activate environment (in new shells):"
            echo "     source venv/bin/activate"
            echo ""
            echo "  5. Configure your training job in config/"
            echo ""
            echo "  6. Run training:"
            echo "     run-training config/your_config.yml"
            echo ""
            echo "Environment variables:"
            echo "  CUDA_HOME: ${pkgs.cudaPackages.cudatoolkit}"
            echo "  LD_LIBRARY_PATH includes CUDA and system libraries"
            echo ""

            export CUDA_HOME="${pkgs.cudaPackages.cudatoolkit}"
            export CUDA_PATH="${pkgs.cudaPackages.cudatoolkit}"

            # Add all necessary libraries to LD_LIBRARY_PATH
            export LD_LIBRARY_PATH="${pkgs.stdenv.cc.cc.lib}/lib:${pkgs.cudaPackages.cudatoolkit}/lib:${pkgs.cudaPackages.cudnn}/lib:${pkgs.linuxPackages.nvidia_x11}/lib:${pkgs.libGL}/lib:${pkgs.glib}/lib:${pkgs.glibc}/lib:${pkgs.xorg.libX11}/lib:${pkgs.xorg.libXext}/lib:$LD_LIBRARY_PATH"

            # Support for Blackwell architecture (sm_120) and others
            export TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6 8.9 9.0 12.0"

            export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:512"

            # Prisma
            export PKG_CONFIG_PATH="${pkgs.openssl.dev}/lib/pkgconfig"
            export PRISMA_SCHEMA_ENGINE_BINARY="${pkgs.prisma-engines_6}/bin/schema-engine"
            export PRISMA_QUERY_ENGINE_BINARY="${pkgs.prisma-engines_6}/bin/query-engine"
            export PRISMA_QUERY_ENGINE_LIBRARY="${pkgs.prisma-engines_6}/lib/libquery_engine.node"
            export PRISMA_FMT_BINARY="${pkgs.prisma-engines_6}/bin/prisma-fmt"

            if [ -d "venv" ]; then
              echo "📦 Virtual environment exists. Activate with: source venv/bin/activate"
            else
              echo "📦 No virtual environment found."
              echo "   For Blackwell GPU: Run 'setup-venv-nightly' (RECOMMENDED)"
              echo "   For older GPUs: Run 'setup-venv-stable'"
            fi
          '';
        };

        packages = {
          setup-venv-nightly = setupVenvNightly;
          setup-venv-stable = setupVenvStable;
          activate-venv = activateVenv;
          run-training = runTraining;
          start-ui = startUI;
        };

        packages.default = setupVenvNightly;
      }
    );
}
