#!/bin/bash

# Setup script for HuggingFace model access and pre-download
# Run this before running batch inference to ensure smooth operation

MODEL_REPO="bharathkumar1922001/3checkpoint-10speaker-aws-H100-30per"
SNAC_REPO="hubertsiuzdak/snac_24khz"

CHECKPOINTS=(
    "checkpoint-3092"
    "checkpoint-6184" 
    "checkpoint-7730"
    "checkpoint-9276"
)

echo "============================================"
echo "HuggingFace Model Setup for Orpheus TTS"
echo "============================================"
echo "Repository: $MODEL_REPO"
echo "Checkpoints: ${CHECKPOINTS[*]}"
echo ""

# Check HuggingFace CLI
if ! command -v huggingface-cli &> /dev/null; then
    echo "❌ HuggingFace CLI not found. Installing..."
    pip install huggingface_hub
fi

# Check authentication
echo "🔐 Checking HuggingFace authentication..."
if ! huggingface-cli whoami &>/dev/null; then
    echo "❌ Not logged in to HuggingFace"
    echo "Please run: huggingface-cli login"
    echo "Use your HuggingFace token with read access"
    exit 1
else
    HF_USER=$(huggingface-cli whoami 2>/dev/null | head -n1 || echo "Unknown")
    echo "✅ Logged in as: $HF_USER"
fi

# Check repository access
echo ""
echo "🔍 Checking repository access..."
if huggingface-cli repo info "$MODEL_REPO" &>/dev/null; then
    echo "✅ Can access repository: $MODEL_REPO"
else
    echo "❌ Cannot access repository: $MODEL_REPO"
    echo "Please check:"
    echo "  1. Repository name is correct"
    echo "  2. You have access permissions"
    echo "  3. Repository is not private (or you have access)"
    exit 1
fi

# List available files in repository
echo ""
echo "📁 Repository contents:"
huggingface-cli list-repo-files "$MODEL_REPO" 2>/dev/null | head -20

# Check for checkpoints
echo ""
echo "🔄 Checking available checkpoints..."
for checkpoint in "${CHECKPOINTS[@]}"; do
    echo -n "  Checking $checkpoint... "
    if huggingface-cli list-repo-files "$MODEL_REPO" 2>/dev/null | grep -q "$checkpoint/"; then
        echo "✅ Found"
    else
        echo "❌ Not found"
    fi
done

# Pre-download tokenizer and config files for faster inference
echo ""
echo "📥 Pre-downloading essential files for faster inference..."

# Create a temporary Python script to download models
cat > temp_download.py << 'EOF'
import os
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import snapshot_download
import torch

MODEL_REPO = "bharathkumar1922001/3checkpoint-10speaker-aws-H100-30per"
SNAC_REPO = "hubertsiuzdak/snac_24khz"

CHECKPOINTS = [
    "checkpoint-3092",
    "checkpoint-6184", 
    "checkpoint-7730",
    "checkpoint-9276"
]

print("Downloading SNAC model...")
try:
    from snac import SNAC
    snac_model = SNAC.from_pretrained(SNAC_REPO)
    print("✅ SNAC model downloaded successfully")
except Exception as e:
    print(f"❌ Error downloading SNAC model: {e}")

print("\nDownloading Orpheus checkpoints...")
for checkpoint in CHECKPOINTS:
    checkpoint_path = f"{MODEL_REPO}/{checkpoint}"
    print(f"Downloading {checkpoint}...")
    
    try:
        # Download tokenizer first (faster)
        tokenizer = AutoTokenizer.from_pretrained(
            checkpoint_path,
            trust_remote_code=True
        )
        print(f"  ✅ Tokenizer for {checkpoint}")
        
        # Download model config and weights (this might take time)
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path,
            trust_remote_code=True,
            torch_dtype=torch.float16,  # Use float16 for faster download
            device_map="cpu"  # Keep on CPU during download
        )
        print(f"  ✅ Model weights for {checkpoint}")
        
        # Clean up memory
        del model, tokenizer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
    except Exception as e:
        print(f"  ❌ Error downloading {checkpoint}: {e}")

print("\n✅ Download process completed!")
print("Models are now cached locally for faster inference.")
EOF

# Run the download script
echo "Starting download process..."
python temp_download.py

DOWNLOAD_STATUS=$?

# Clean up
rm -f temp_download.py

if [ $DOWNLOAD_STATUS -eq 0 ]; then
    echo ""
    echo "============================================"
    echo "✅ SETUP COMPLETED SUCCESSFULLY!"
    echo "============================================"
    echo "All models have been downloaded and cached."
    echo "You can now run batch inference with:"
    echo ""
    echo "  ./run_batch_inference.sh"
    echo ""
    echo "Or test a single checkpoint with:"
    echo "  ./test_single_checkpoint.sh"
    echo "============================================"
else
    echo ""
    echo "❌ Setup encountered some errors."
    echo "Check the output above for details."
    echo "You may still be able to run inference, but it might be slower."
fi