#!/bin/bash

# Single Checkpoint Test Script
# Test one checkpoint with 2 speakers and 2 sentences to validate setup

MODEL_BASE_PATH="bharathkumar1922001/3checkpoint-10speaker-aws-H100-30per"
CHECKPOINT="checkpoint-3092"  # Change this to test different checkpoints
OUTPUT_DIR="test_inference_$(date +%Y%m%d_%H%M%S)"

echo "================================"
echo "Single Checkpoint Test"
echo "================================"
echo "Testing: ${MODEL_BASE_PATH}/${CHECKPOINT}"
echo "Output: $OUTPUT_DIR"
echo ""

# Check GPU
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Status:"
    nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader,nounits | head -1
    echo ""
fi

echo "Running test inference..."
python enhanced_infer.py \
    --single_checkpoint "${MODEL_BASE_PATH}/${CHECKPOINT}" \
    --output_dir "$OUTPUT_DIR"

echo ""
echo "Test completed! Check $OUTPUT_DIR for results."
echo ""
echo "Expected structure:"
echo "$OUTPUT_DIR/"
echo "└── $CHECKPOINT/"
echo "    ├── aisha/"
echo "    │   ├── aisha_sentence_01_*.wav"
echo "    │   └── ..."
echo "    ├── anika/"
echo "    └── ..."
echo ""

# Quick validation
python -c "
import os
import glob

output_dir = '$OUTPUT_DIR'
checkpoint = '$CHECKPOINT'
expected_files = 10 * 5  # 10 speakers × 5 sentences

checkpoint_dir = os.path.join(output_dir, checkpoint)
if os.path.exists(checkpoint_dir):
    wav_files = glob.glob(os.path.join(checkpoint_dir, '**', '*.wav'), recursive=True)
    print(f'Generated files: {len(wav_files)}/{expected_files}')
    if len(wav_files) > 0:
        print('✓ Test successful! Ready for batch inference.')
        # Show first few files
        print('\nSample files:')
        for f in sorted(wav_files)[:3]:
            print(f'  - {os.path.basename(f)}')
        if len(wav_files) > 3:
            print(f'  ... and {len(wav_files)-3} more')
    else:
        print('✗ No files generated. Check for errors above.')
else:
    print('✗ Output directory not found. Check for errors above.')
"