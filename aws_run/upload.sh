#!/bin/bash

# Dataset Upload Runner for Orpheus TTS Audio Results
# This script uploads your batch inference results to Hugging Face as an audio dataset

# Configuration
DEFAULT_REPO_NAME="orpheus-hindi-tts-comparison-$(date +%Y%m%d)"
INPUT_DIR=""
REPO_NAME=""
PRIVATE_FLAG=""
CLEANUP_FLAG="--cleanup"

echo "============================================"
echo "Orpheus TTS Dataset Uploader"
echo "============================================"

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -i, --input-dir DIR        Directory containing batch inference results (required)"
    echo "  -r, --repo-name NAME       HuggingFace dataset repository name"
    echo "  -p, --private              Make dataset private"
    echo "  -k, --keep-temp            Keep temporary files (don't cleanup)"
    echo "  -h, --help                 Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 -i batch_inference_results_20241224_143022 -r username/orpheus-hindi-comparison"
    echo "  $0 -i ./my_results -r bharathkumar1922001/orpheus-tts-eval -p"
    echo ""
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -i|--input-dir)
            INPUT_DIR="$2"
            shift 2
            ;;
        -r|--repo-name)
            REPO_NAME="$2"
            shift 2
            ;;
        -p|--private)
            PRIVATE_FLAG="--private"
            shift
            ;;
        -k|--keep-temp)
            CLEANUP_FLAG=""
            shift
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Validate required arguments
if [[ -z "$INPUT_DIR" ]]; then
    echo "Error: Input directory is required"
    echo ""
    show_usage
    exit 1
fi

if [[ ! -d "$INPUT_DIR" ]]; then
    echo "Error: Input directory does not exist: $INPUT_DIR"
    exit 1
fi

# Set default repo name if not provided
if [[ -z "$REPO_NAME" ]]; then
    REPO_NAME="$DEFAULT_REPO_NAME"
    echo "Using default repository name: $REPO_NAME"
    echo "You can specify a custom name with -r option"
    echo ""
fi

# Check for audio files in input directory
AUDIO_COUNT=$(find "$INPUT_DIR" -name "*.wav" | wc -l)
if [[ $AUDIO_COUNT -eq 0 ]]; then
    echo "Error: No WAV files found in $INPUT_DIR"
    echo "Make sure you're pointing to the correct batch inference results directory"
    exit 1
fi

echo "Configuration:"
echo "  Input Directory: $INPUT_DIR"
echo "  Audio Files Found: $AUDIO_COUNT"
echo "  Repository Name: $REPO_NAME"
echo "  Private: $([ -n "$PRIVATE_FLAG" ] && echo "Yes" || echo "No")"
echo "  Cleanup Temp: $([ -n "$CLEANUP_FLAG" ] && echo "Yes" || echo "No")"
echo ""

# Check HuggingFace CLI login
echo "Checking HuggingFace authentication..."
if ! huggingface-cli whoami &>/dev/null; then
    echo "❌ Not logged in to HuggingFace"
    echo "Please run: huggingface-cli login"
    echo "Then re-run this script"
    exit 1
else
    HF_USER=$(huggingface-cli whoami | head -n1)
    echo "✅ Logged in as: $HF_USER"
fi

# Check if repo name includes username
if [[ "$REPO_NAME" != *"/"* ]]; then
    echo "Note: Repository name should include username (e.g., username/dataset-name)"
    echo "Current: $REPO_NAME"
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo ""
echo "Starting dataset creation and upload..."
echo "This may take a few minutes depending on the number of audio files..."
echo ""

# Run the Python script
python push_to_hf_dataset.py \
    --input_dir "$INPUT_DIR" \
    --repo_name "$REPO_NAME" \
    $PRIVATE_FLAG \
    $CLEANUP_FLAG

UPLOAD_STATUS=$?

if [[ $UPLOAD_STATUS -eq 0 ]]; then
    echo ""
    echo "============================================"
    echo "✅ DATASET UPLOAD SUCCESSFUL!"
    echo "============================================"
    echo "🎵 Dataset URL: https://huggingface.co/datasets/$REPO_NAME"
    echo ""
    echo "What you can do now:"
    echo "1. Visit the dataset page to explore your audio files"
    echo "2. Use the Dataset Viewer to listen and compare samples"
    echo "3. Filter samples by:"
    echo "   - Speaker (aisha, anika, arfa, etc.)"
    echo "   - Checkpoint (checkpoint-3092, checkpoint-6184, etc.)"
    echo "   - Sentence category (Business, Casual, Emotional, etc.)"
    echo ""
    echo "Load in Python:"
    echo "  from datasets import load_dataset"
    echo "  dataset = load_dataset('$REPO_NAME')"
    echo "============================================"
else
    echo ""
    echo "❌ Dataset upload failed. Check the error messages above."
    exit 1
fi