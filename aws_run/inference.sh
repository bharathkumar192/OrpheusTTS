#!/bin/bash

# Batch Inference Runner for Orpheus TTS Model
# This script runs inference on all 4 checkpoints with all 10 speakers and 5 Hindi sentences

# Configuration
MODEL_BASE_PATH="bharathkumar1922001/orpheus-10speakers-8xH100-oom-fixed-v1"  # Your HuggingFace model ID
OUTPUT_DIR="batch_inference_results_$(date +%Y%m%d_%H%M%S)"
MAX_PARALLEL=2  # Number of checkpoints to process simultaneously

# Checkpoint names
CHECKPOINTS=(
    "checkpoint-3092"
    "checkpoint-6184" 
    "checkpoint-7730"
    "checkpoint-9276"
)

echo "=================================="
echo "Orpheus TTS Batch Inference Runner"
echo "=================================="
echo "Model: $MODEL_BASE_PATH"
echo "Output Directory: $OUTPUT_DIR"
echo "Checkpoints: ${CHECKPOINTS[*]}"
echo "Max Parallel: $MAX_PARALLEL"
echo ""

# Create checkpoint paths
CHECKPOINT_PATHS=""
for checkpoint in "${CHECKPOINTS[@]}"; do
    CHECKPOINT_PATHS="$CHECKPOINT_PATHS ${MODEL_BASE_PATH}/${checkpoint}"
done

echo "Full checkpoint paths:"
for checkpoint in "${CHECKPOINTS[@]}"; do
    echo "  - ${MODEL_BASE_PATH}/${checkpoint}"
done
echo ""

# Check if GPU is available
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Information:"
    nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader,nounits
    echo ""
else
    echo "Warning: nvidia-smi not found. Running on CPU."
    echo ""
fi

# Run the batch inference
echo "Starting batch inference..."
echo "Expected output: 200 audio files (10 speakers × 5 sentences × 4 checkpoints)"
echo ""

python enhanced_infer.py \
    --checkpoint_paths $CHECKPOINT_PATHS \
    --output_dir "$OUTPUT_DIR" \
    --max_parallel $MAX_PARALLEL

echo ""
echo "=================================="
echo "Batch inference completed!"
echo "Results saved in: $OUTPUT_DIR"
echo ""
echo "Directory structure:"
echo "$OUTPUT_DIR/"
echo "├── checkpoint-3092/"
echo "│   ├── aisha/"
echo "│   │   ├── aisha_sentence_01_*.wav"
echo "│   │   ├── aisha_sentence_02_*.wav"
echo "│   │   └── ..."
echo "│   ├── anika/"
echo "│   └── ..."
echo "├── checkpoint-6184/"
echo "├── checkpoint-7730/"
echo "└── checkpoint-9276/"
echo "=================================="

# Generate summary report
echo "Generating summary report..."
python -c "
import os
import json
import glob
from pathlib import Path

output_dir = '$OUTPUT_DIR'
checkpoints = ['checkpoint-3092', 'checkpoint-6184', 'checkpoint-7730', 'checkpoint-9276']
speakers = ['aisha', 'anika', 'arfa', 'asmr', 'nikita', 'raju', 'rhea', 'ruhaan', 'sangeeta', 'shayana']

total_expected = len(checkpoints) * len(speakers) * 5
total_generated = 0
checkpoint_stats = {}

print('\n=== BATCH INFERENCE SUMMARY ===')
for checkpoint in checkpoints:
    checkpoint_dir = os.path.join(output_dir, checkpoint)
    if os.path.exists(checkpoint_dir):
        wav_files = glob.glob(os.path.join(checkpoint_dir, '**', '*.wav'), recursive=True)
        checkpoint_stats[checkpoint] = len(wav_files)
        total_generated += len(wav_files)
        print(f'{checkpoint}: {len(wav_files)}/50 files')
    else:
        checkpoint_stats[checkpoint] = 0
        print(f'{checkpoint}: 0/50 files (directory not found)')

print(f'\nTotal generated: {total_generated}/{total_expected} files')
print(f'Success rate: {(total_generated/total_expected)*100:.1f}%')

# Check for any summary JSON files
summary_files = glob.glob(os.path.join(output_dir, '**', '*_summary.json'), recursive=True)
if summary_files:
    print(f'\nDetailed summaries available in:')
    for sf in summary_files:
        print(f'  - {sf}')
"

echo ""
echo "Summary report generated!"
echo "Check the output directory for detailed results and any error logs."