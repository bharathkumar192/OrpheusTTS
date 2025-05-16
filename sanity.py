import os
import argparse
from datasets import load_dataset, ClassLabel
from transformers import AutoTokenizer
import random
from collections import Counter

# --- Configuration ---
PROCESSED_DATASET_ID = "bharathkumar1922001/aisha-asmr-zadok-anika-ivanna-raju-sia-sangeeta-"  # UPDATE THIS!

# Tokenizer used during preprocessing (for decoding text segments)
ORPHEUS_TOKENIZER_NAME = "canopylabs/3b-hi-pretrain-research_release"

# Number of random samples to check in detail
NUM_SAMPLES_TO_INSPECT = 10

# Expected columns in the processed dataset
EXPECTED_COLUMNS = ["input_ids", "labels", "attention_mask"]
# Optional columns that might be present for inspection
OPTIONAL_COLUMNS = ["speaker_id", "sentence_id"]


# --- Special Tokens (from preprocess.py, ensure consistency) ---
tokeniser_length = 128256
start_of_speech_token = tokeniser_length + 1
end_of_speech_token = tokeniser_length + 2
start_of_human_token = tokeniser_length + 3
end_of_human_token = tokeniser_length + 4
start_of_ai_token = tokeniser_length + 5
end_of_ai_token = tokeniser_length + 6

SPECIAL_TOKENS_MAP_CHECK = {
    start_of_speech_token: "<SOS>", end_of_speech_token: "<EOSpeech>",
    start_of_human_token: "<SOH>", end_of_human_token: "<EOH>",
    start_of_ai_token: "<SOA>", end_of_ai_token: "<EOA>",
}
ALL_SPECIAL_TOKEN_IDS = set(SPECIAL_TOKENS_MAP_CHECK.keys())


def verify_sample_structure(sample, tokenizer, sample_idx="N/A"):
    """
    Performs detailed checks on a single sample.
    Returns a list of error/warning messages.
    """
    errors = []
    warnings = []

    input_ids = sample.get("input_ids")
    labels = sample.get("labels")
    attention_mask = sample.get("attention_mask")
    speaker_id = sample.get("speaker_id") # Optional

    # Basic presence and type checks
    if not isinstance(input_ids, list) or not all(isinstance(i, int) for i in input_ids):
        errors.append(f"Sample {sample_idx}: 'input_ids' is not a list of integers.")
    if not isinstance(labels, list) or not all(isinstance(i, int) for i in labels):
        errors.append(f"Sample {sample_idx}: 'labels' is not a list of integers.")
    if not isinstance(attention_mask, list) or not all(isinstance(i, int) for i in attention_mask):
        errors.append(f"Sample {sample_idx}: 'attention_mask' is not a list of integers.")

    if errors: return errors, warnings # Stop if fundamental types are wrong

    # Length checks
    len_input = len(input_ids)
    if len_input == 0:
        errors.append(f"Sample {sample_idx}: 'input_ids' is empty.")
        return errors, warnings
    if len(labels) != len_input:
        errors.append(f"Sample {sample_idx}: Length mismatch between 'input_ids' ({len_input}) and 'labels' ({len(labels)}).")
    if len(attention_mask) != len_input:
        errors.append(f"Sample {sample_idx}: Length mismatch between 'input_ids' ({len_input}) and 'attention_mask' ({len(attention_mask)}).")

    # Label check (should be identical to input_ids before collator)
    if input_ids != labels:
        # Find first mismatch for reporting
        mismatch_idx = -1
        for k, (i_val, l_val) in enumerate(zip(input_ids, labels)):
            if i_val != l_val:
                mismatch_idx = k
                break
        if mismatch_idx != -1:
            errors.append(f"Sample {sample_idx}: 'input_ids' and 'labels' are not identical. First mismatch at index {mismatch_idx}: input={input_ids[mismatch_idx]}, label={labels[mismatch_idx]}.")
        else: # Should not happen if lengths are same and they differ
             errors.append(f"Sample {sample_idx}: 'input_ids' and 'labels' are not identical (general mismatch).")


    # Attention mask check (should be all 1s before collator)
    if not all(bit == 1 for bit in attention_mask):
        warnings.append(f"Sample {sample_idx}: 'attention_mask' contains zeros. Sum: {sum(attention_mask)}/{len_input}. This is unexpected for uncollated data.")

    # Special token structure check
    try:
        soh_idx = input_ids.index(start_of_human_token)
        eoh_idx = input_ids.index(end_of_human_token)
        soa_idx = input_ids.index(start_of_ai_token)
        sos_idx = input_ids.index(start_of_speech_token)
        eos_idx = input_ids.index(end_of_speech_token)
        eoa_idx = input_ids.index(end_of_ai_token)

        # Order check (basic)
        if not (soh_idx < eoh_idx < soa_idx < sos_idx < eos_idx < eoa_idx):
            errors.append(f"Sample {sample_idx}: Special tokens are out of expected order.")

        # Text segment decoding and speaker prefix check
        if speaker_id and (soh_idx < eoh_idx):
            text_tokens = input_ids[soh_idx+1 : eoh_idx]
            # Filter out any other special tokens that might have slipped into the text segment IF NECESSARY
            # text_tokens_for_decode = [t for t in text_tokens if t not in ALL_SPECIAL_TOKEN_IDS]
            # For now, assume text segment is clean from preprocess.py
            decoded_text = tokenizer.decode(text_tokens, skip_special_tokens=True) # skip_special_tokens=True to clean up for readability
            
            expected_prefix = f"{speaker_id}:"
            if not decoded_text.startswith(expected_prefix):
                warnings.append(f"Sample {sample_idx} (Speaker: {speaker_id}): Decoded text segment does not start with expected prefix '{expected_prefix}'. Found: '{decoded_text[:len(expected_prefix)+20]}...'")
        elif not speaker_id:
            warnings.append(f"Sample {sample_idx}: 'speaker_id' column missing or empty in sample, cannot verify text prefix.")

    except ValueError as e: # If any .index() fails
        errors.append(f"Sample {sample_idx}: Could not find one or more required special tokens in 'input_ids'. Missing token related to: {e}")
    except Exception as e:
        errors.append(f"Sample {sample_idx}: Unexpected error during special token structure check: {e}")

    return errors, warnings


def main():
    parser = argparse.ArgumentParser(description="Perform sanity checks on a processed Hugging Face dataset.")
    parser.add_argument(
        "--dataset_id",
        type=str,
        default=PROCESSED_DATASET_ID,
        help="Hugging Face Hub ID of the processed dataset to check."
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=ORPHEUS_TOKENIZER_NAME,
        help="Name or path of the tokenizer used for preprocessing."
    )
    parser.add_argument(
        "--num_inspect",
        type=int,
        default=NUM_SAMPLES_TO_INSPECT,
        help="Number of random samples to inspect in detail."
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to check (e.g., 'train')."
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="Hugging Face API token for private datasets."
    )

    args = parser.parse_args()

    print(f"--- Sanity Check Initializing ---")
    print(f"Processed Dataset ID: {args.dataset_id}")
    print(f"Tokenizer: {args.tokenizer_name}")
    print(f"Split to check: {args.split}")
    print(f"Number of samples to inspect: {args.num_inspect}")
    print("-" * 30)

    # --- Load Tokenizer ---
    print(f"\n--- Loading Tokenizer ({args.tokenizer_name}) ---")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, token=args.hf_token)
        print("Tokenizer loaded successfully.")
    except Exception as e:
        print(f"ERROR: Could not load tokenizer '{args.tokenizer_name}': {e}")
        exit(1)

    # --- Load Dataset ---
    print(f"\n--- Loading Processed Dataset ({args.dataset_id}, split: {args.split}) ---")
    try:
        dataset = load_dataset(args.dataset_id, split=args.split, token=args.hf_token, trust_remote_code=True)
        print(f"Dataset loaded successfully with {len(dataset)} samples.")
    except Exception as e:
        print(f"ERROR: Could not load dataset '{args.dataset_id}': {e}")
        if "Could not find a ClassLabel from string" in str(e) or "Features" in str(e):
            print("Hint: This might be due to inconsistencies in how features were defined during dataset push vs. loading.")
            print("Ensure 'input_ids', 'labels', 'attention_mask' are sequences of integers, and 'speaker_id' (if present) is string.")
        exit(1)

    # --- 1. Column Checks ---
    print(f"\n--- 1. Column Checks ---")
    actual_columns = dataset.column_names
    print(f"Actual columns found: {actual_columns}")
    missing_expected = [col for col in EXPECTED_COLUMNS if col not in actual_columns]
    if missing_expected:
        print(f"ERROR: Missing expected columns: {missing_expected}")
        exit(1)
    else:
        print("All expected columns are present.")
    
    present_optional = [col for col in OPTIONAL_COLUMNS if col in actual_columns]
    if present_optional:
        print(f"Found optional columns: {present_optional}")
    
    # --- 2. Data Type and Basic Structure Checks (on first few samples) ---
    print(f"\n--- 2. Data Type and Basic Structure Checks (on first 3 samples) ---")
    for i in range(min(3, len(dataset))):
        sample = dataset[i]
        print(f"  Sample {i}:")
        for col in EXPECTED_COLUMNS:
            val = sample.get(col)
            print(f"    Column '{col}': Type={type(val).__name__}, Len={len(val) if isinstance(val, list) else 'N/A'}")
            if not isinstance(val, list):
                print(f"    ERROR: Column '{col}' in sample {i} is not a list.")
            elif val and not isinstance(val[0], int):
                 print(f"    ERROR: Column '{col}' in sample {i} is not a list of integers (first element type: {type(val[0]).__name__}).")
        if 'speaker_id' in sample:
             print(f"    Column 'speaker_id': Type={type(sample['speaker_id']).__name__}, Value='{str(sample['speaker_id'])[:50]}'")
             if not isinstance(sample['speaker_id'], str):
                 print(f"    ERROR: Column 'speaker_id' in sample {i} is not a string.")


    # --- 3. Detailed Sample Inspection ---
    print(f"\n--- 3. Detailed Inspection of {args.num_inspect} Random Samples ---")
    if len(dataset) == 0:
        print("Dataset is empty, skipping detailed inspection.")
    elif args.num_inspect == 0:
        print("Number of samples to inspect is 0, skipping.")
    else:
        num_to_pick = min(args.num_inspect, len(dataset))
        random_indices = random.sample(range(len(dataset)), num_to_pick)
        
        total_errors_detail = 0
        total_warnings_detail = 0

        for i, sample_idx in enumerate(random_indices):
            print(f"\nInspecting Sample #{i+1} (Dataset Index: {sample_idx})")
            sample = dataset[sample_idx]
            
            # Display basic info for context
            if "speaker_id" in sample: print(f"  Speaker ID: {sample['speaker_id']}")
            if "sentence_id" in sample: print(f"  Sentence ID: {sample['sentence_id']}")
            print(f"  Input IDs length: {len(sample.get('input_ids', []))}")
            # print(f"  Input IDs (first 10): {sample.get('input_ids', [])[:10]}") # Usually too verbose here

            errors, warnings = verify_sample_structure(sample, tokenizer, sample_idx=sample_idx)
            
            if errors:
                print(f"  ERRORS found for sample {sample_idx}:")
                for err in errors: print(f"    - {err}")
                total_errors_detail += len(errors)
            if warnings:
                print(f"  WARNINGS found for sample {sample_idx}:")
                for warn in warnings: print(f"    - {warn}")
                total_warnings_detail += len(warnings)
            
            if not errors and not warnings:
                print(f"  Sample {sample_idx} PASSED detailed structure checks.")
        
        print(f"\nDetailed inspection summary: {total_errors_detail} errors, {total_warnings_detail} warnings found across {num_to_pick} samples.")
        if total_errors_detail > 0:
            print("CRITICAL: Errors found during detailed sample inspection. Review output above.")


    # --- 4. Speaker ID Distribution (if 'speaker_id' column exists) ---
    if 'speaker_id' in dataset.column_names:
        print(f"\n--- 4. Speaker ID Distribution ---")
        try:
            speaker_ids_list = dataset['speaker_id'] # This loads the whole column, can be memory intensive for huge datasets
            speaker_counts = Counter(speaker_ids_list)
            num_unique_speakers = len(speaker_counts)
            print(f"Found {num_unique_speakers} unique speaker IDs.")
            
            if num_unique_speakers > 0:
                print("\nTop 10 speakers by sample count:")
                for speaker, count in speaker_counts.most_common(10):
                    print(f"  - Speaker '{speaker}': {count} samples")
                
                counts_values = list(speaker_counts.values())
                print(f"\nSample counts per speaker stats:")
                print(f"  Min samples per speaker: {min(counts_values)}")
                print(f"  Max samples per speaker: {max(counts_values)}")
                print(f"  Avg samples per speaker: {sum(counts_values) / num_unique_speakers:.2f}")

                if num_unique_speakers < 2 and len(dataset) > 0: # Check if it's truly multi-speaker
                     warnings.append(f"Speaker Check: Only {num_unique_speakers} unique speaker ID found. Expected a multi-speaker dataset.")
                     print(f"WARNING: Only {num_unique_speakers} unique speaker ID found. Expected a multi-speaker dataset.")


        except Exception as e:
            print(f"Could not compute speaker ID distribution: {e}")
            print("This might happen if 'speaker_id' column is not a simple list of strings or due to memory issues for very large datasets.")
    else:
        print("\n--- 4. Speaker ID Distribution ---")
        print("'speaker_id' column not found, skipping distribution analysis.")
        if len(dataset) > 0: # If dataset has content but no speaker_id, it's a problem for multi-speaker
            warnings.append("Speaker Check: 'speaker_id' column is missing, which is essential for multi-speaker setup.")
            print("CRITICAL WARNING: 'speaker_id' column is missing, but this is expected to be a multi-speaker dataset.")


    # --- Final Summary ---
    print(f"\n--- Sanity Check Finished ---")
    if total_errors_detail > 0 or any("ERROR:" in line for line in printed_lines): # Heuristic for earlier errors
        print("RESULT: One or more CRITICAL ERRORS found. Please review the logs carefully before training.")
    elif total_warnings_detail > 0 or any("WARNING:" in line for line in printed_lines):
        print("RESULT: Some WARNINGS found. Review the logs. Training might be possible but proceed with caution.")
    else:
        print("RESULT: All basic checks passed, and no errors/warnings in detailed sample inspection.")
        print("The dataset appears to be structurally sound for training.")

if __name__ == "__main__":
    # Keep track of printed lines to check for earlier errors in summary (simple heuristic)
    printed_lines = []
    original_print = print
    def new_print(*args, **kwargs):
        global printed_lines
        line = " ".join(str(arg) for arg in args)
        printed_lines.append(line)
        original_print(*args, **kwargs)
    print = new_print
    main()