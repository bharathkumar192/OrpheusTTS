#!/bin/bash

echo "Setting up environment..."

# Install required Python packages
echo "Installing Python packages..."
pip install transformers datasets wandb trl accelerate torch huggingface_hub snac wave numpy librosa soundfile orpheus-speech "vllm>=0.4.0"


# For faster training (requires compatible GPU and CUDA toolkit)
pip3 install flash-attn --no-build-isolation
apt update && sudo apt install git-lfs && git lfs install --system


# Log in to Wandb
echo "Logging in to Wandb..."
wandb login <token>

# Log in to Huggingface
echo "Logging in to Huggingface..."
# huggingface-cli login 
huggingface-cli login --token <token>  --add-to-git-credential
echo "Setup complete!" 
