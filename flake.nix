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

        setupVenv = pkgs.writeShellScriptBin "setup-venv" ''
          set -e
          
          if [ ! -d "venv" ]; then
            echo "Creating Python virtual environment..."
            ${python}/bin/python -m venv venv
            
            echo "Activating virtual environment..."
            source venv/bin/activate
            
            echo "Upgrading pip..."
            pip install --upgrade pip setuptools wheel
            
            echo "Installing PyTorch with CUDA support..."
            pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
            
            echo "Installing requirements from requirements.txt..."
            if [ -f "requirements.txt" ]; then
              pip install -r requirements.txt
            else
              echo "Warning: requirements.txt not found. Installing common dependencies..."
              pip install accelerate transformers diffusers huggingface_hub
              pip install safetensors omegaconf pillow tqdm pyyaml
              pip install bitsandbytes triton xformers
            fi
            
            echo "Virtual environment setup complete!"
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
            echo "Run 'setup-venv' first."
            exit 1
          fi
        '';

        runTraining = pkgs.writeShellScriptBin "run-training" ''
          if [ -f "venv/bin/activate" ]; then
            source venv/bin/activate
            python run.py "$@"
          else
            echo "Error: Virtual environment not found."
            echo "Run 'setup-venv' first."
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
            cd ui
            ${pkgs.nodejs}/bin/npm run build_and_start
          else
            echo "Error: Virtual environment not found."
            echo "Run 'setup-venv' first."
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

          zlib
          libjpeg
          libpng
          libtiff

          mkl
        ];

      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = systemDeps ++ [
            setupVenv
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
            echo "  setup-venv      - Create and setup Python virtual environment"
            echo "  activate-venv   - Activate the virtual environment"
            echo "  run-training    - Run training with config file"
            echo "  start-ui        - Start the web UI"
            echo ""
            echo "Quick start:"
            echo "  1. Clone the repository if you haven't:"
            echo "     git clone https://github.com/ostris/ai-toolkit.git"
            echo "     cd ai-toolkit"
            echo ""
            echo "  2. Initialize submodules:"
            echo "     git submodule update --init --recursive"
            echo ""
            echo "  3. Setup Python environment:"
            echo "     setup-venv"
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
            echo "  LD_LIBRARY_PATH includes CUDA libraries"
            echo ""

            export CUDA_HOME="${pkgs.cudaPackages.cudatoolkit}"
            export CUDA_PATH="${pkgs.cudaPackages.cudatoolkit}"

            export LD_LIBRARY_PATH="${pkgs.cudaPackages.cudatoolkit}/lib:${pkgs.cudaPackages.cudnn}/lib:${pkgs.linuxPackages.nvidia_x11}/lib:$LD_LIBRARY_PATH"

            export TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6 8.9 9.0"

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
              echo "📦 No virtual environment found. Run: setup-venv"
            fi
          '';
        };

        packages = {
          setup-venv = setupVenv;
          activate-venv = activateVenv;
          run-training = runTraining;
          start-ui = startUI;
        };

        packages.default = setupVenv;
      }
    );
}