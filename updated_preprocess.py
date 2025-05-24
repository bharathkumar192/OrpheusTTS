import os
import torch
import torchaudio.transforms as T
from datasets import load_dataset, Dataset, Features, Value, Audio, concatenate_datasets, Sequence
from huggingface_hub import HfApi, upload_folder # Added upload_folder
from snac import SNAC
from transformers import AutoTokenizer
import random
import json
import numpy as np
# from tqdm import tqdm # tqdm can be verbose with batched map, consider removing if not essential for progress
import yaml
import multiprocessing
import time
import gc
import argparse
from collections import Counter # For vectorized counting

# --- Configuration ---
CONFIG_FILE_PATH_FOR_PREPROCESS = "config.yaml"
DATASET_REPO_ID = "maya-research/Multi-speaker-TTS" # Default, can be overridden by args
# LOCAL_RAW_DATASET_DIR = os.path.join(os.getcwd(), "downloaded_dataset_raw") # Not used if loading from HF
PROCESSED_HF_DATASET_REPO_FOR_UPLOAD = "bharathkumar1922001/2-speaker-shayana-raju" # Base name, may be modified

ORPHEUS_TOKENIZER_NAME = "canopylabs/3b-hi-pretrain-research_release"
SNAC_MODEL_NAME = "hubertsiuzdak/snac_24khz"
TARGET_AUDIO_SAMPLING_RATE = 24000

# --- Performance Tuning Globals (from suggestions) ---
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
OPTIMIZED_NUM_PROC = min(16, multiprocessing.cpu_count()) # For CPU-bound tasks
OPTIMIZED_BATCH_SIZE = 50_000 # For batched .map() operations
OPTIMIZED_MAX_SHARD_SIZE = "2GB" # For push_to_hub

# Special Token IDs
tokeniser_length = 128256
start_of_speech_token = tokeniser_length + 1
end_of_speech_token = tokeniser_length + 2
start_of_human_token = tokeniser_length + 3
end_of_human_token = tokeniser_length + 4
start_of_ai_token = tokeniser_length + 5
end_of_ai_token = tokeniser_length + 6
AUDIO_CODE_BASE_OFFSET = tokeniser_length + 10

# --- Globals ---
text_tokenizer = None
device = None
snac_model = None


def get_speakers_to_filter_from_config(config_path):
    # ... (Your existing function is fine, no changes needed here based on suggestions)
    speakers = []
    try:
        abs_config_path = config_path
        if not os.path.exists(abs_config_path):
             abs_config_path = os.path.join(os.getcwd(), config_path)

        if os.path.exists(abs_config_path):
            with open(abs_config_path, "r") as f: config_data = yaml.safe_load(f)
            speakers = config_data.get("speakers_to_train", [])
            if speakers: print(f"Read 'speakers_to_train' from {abs_config_path}: {speakers}")
            else: print(f"'speakers_to_train' not found/empty in {abs_config_path}. Will process all.")
        else: print(f"Config file {config_path} not found at {abs_config_path}. Will process all speakers.")
    except Exception as e: print(f"Error reading config {config_path}: {e}. Will process all.")
    return speakers if isinstance(speakers, list) else []

def load_models_and_tokenizer():
    global text_tokenizer, snac_model, device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")

    print(f"\n--- Loading Orpheus Text Tokenizer ({ORPHEUS_TOKENIZER_NAME}) ---")
    try:
        text_tokenizer = AutoTokenizer.from_pretrained(ORPHEUS_TOKENIZER_NAME)
        print("Orpheus text tokenizer loaded.")
    except Exception as e: print(f"ERROR loading Orpheus text tokenizer: {e}"); exit(1)

    print(f"\n--- Loading SNAC Model ({SNAC_MODEL_NAME}) ---")
    try:
        snac_model = SNAC.from_pretrained(SNAC_MODEL_NAME).to(device).eval()
        print(f"SNAC model loaded to {device} and set to eval mode.")
    except Exception as e: print(f"ERROR loading SNAC model: {e}"); exit(1)

def tokenize_audio_snac(audio_data_dict):
    # ... (Your existing function is fine for row-by-row, as SNAC is GPU-bound, less impact from batching here)
    # This function is called by add_snac_codes_map_fn which uses num_proc=1 (GPU task)
    global snac_model, device
    if snac_model is None or device is None:
         raise RuntimeError("SNAC model or device not initialized. Call load_models_and_tokenizer first.")
    if not audio_data_dict or 'array' not in audio_data_dict or 'sampling_rate' not in audio_data_dict: return None
    waveform_np = audio_data_dict["array"]; original_sr = audio_data_dict["sampling_rate"]

    if waveform_np is None or len(waveform_np) == 0: return None

    try:
        waveform = torch.from_numpy(waveform_np).unsqueeze(0).to(dtype=torch.float32)
        if waveform.dim() == 1: waveform = waveform.unsqueeze(0)

        waveform = waveform.to(device)
        if original_sr != TARGET_AUDIO_SAMPLING_RATE:
            resample_transform = T.Resample(orig_freq=original_sr, new_freq=TARGET_AUDIO_SAMPLING_RATE).to(device)
            waveform = resample_transform(waveform)

        if waveform.dim() == 2: waveform = waveform.unsqueeze(0)

        with torch.inference_mode():
            codes = snac_model.encode(waveform)

        if not isinstance(codes, list) or len(codes) < 3: return None

        codes_lvl0_cpu = codes[0].squeeze(1).cpu()
        codes_lvl1_cpu = codes[1].squeeze(1).cpu()
        codes_lvl2_cpu = codes[2].squeeze(1).cpu()

        num_coarse_frames = codes_lvl0_cpu.shape[1]
        if num_coarse_frames == 0: return None

        required_len_lvl1 = 2 * num_coarse_frames
        required_len_lvl2 = 4 * num_coarse_frames
        if codes_lvl1_cpu.shape[1] < required_len_lvl1 or codes_lvl2_cpu.shape[1] < required_len_lvl2:
             return None

        all_snac_tokens = []
        for i in range(num_coarse_frames):
                all_snac_tokens.extend([
                    codes_lvl0_cpu[0, i].item() + AUDIO_CODE_BASE_OFFSET,
                    codes_lvl1_cpu[0, 2*i].item() + AUDIO_CODE_BASE_OFFSET + 4096,
                    codes_lvl2_cpu[0, 4*i].item() + AUDIO_CODE_BASE_OFFSET + (2*4096),
                    codes_lvl2_cpu[0, (4*i)+1].item() + AUDIO_CODE_BASE_OFFSET + (3*4096),
                    codes_lvl1_cpu[0, (2*i)+1].item() + AUDIO_CODE_BASE_OFFSET + (4*4096),
                    codes_lvl2_cpu[0, (4*i)+2].item() + AUDIO_CODE_BASE_OFFSET + (5*4096),
                    codes_lvl2_cpu[0, (4*i)+3].item() + AUDIO_CODE_BASE_OFFSET + (6*4096)])
        return all_snac_tokens
    except Exception as e: # Minimal print to avoid stdout bottleneck if this were parallelized on CPU
        # print(f"Error in tokenize_audio_snac: {e}") # Keep for debugging if needed, but remove for prod parallel runs
        return None

def add_snac_codes_map_fn(example): # This remains row-by-row as it's GPU bound
    try:
        audio_data = example.get("audio")
        snac_tokens = tokenize_audio_snac(audio_data)
        example["snac_codes"] = snac_tokens
    except Exception as e:
        example["snac_codes"] = None
    return example

def remove_duplicate_frames_batched_map_fn(examples_batch): # Renamed for clarity
    # This is already batched, so the structure is good.
    # The `examples_batch` is a dict of lists (columns).
    results_batch_list = []
    for snac_tokens_list_item in examples_batch['snac_codes']: # Iterate through the list of snac_codes lists
        if not isinstance(snac_tokens_list_item, list) or not snac_tokens_list_item or len(snac_tokens_list_item) % 7 != 0:
            results_batch_list.append(None)
            continue
        try:
            if len(snac_tokens_list_item) < 7:
                 results_batch_list.append(snac_tokens_list_item if snac_tokens_list_item else None)
                 continue
            result_tokens_single_item = snac_tokens_list_item[:7]
            for i in range(7, len(snac_tokens_list_item), 7):
                if snac_tokens_list_item[i] != result_tokens_single_item[-7]:
                    result_tokens_single_item.extend(snac_tokens_list_item[i:i+7])
            results_batch_list.append(result_tokens_single_item if result_tokens_single_item else None)
        except Exception as e:
             results_batch_list.append(None)
    return {"snac_codes_deduped": results_batch_list}


def create_final_input_batched_map_fn(examples_batch, # Renamed for clarity
                              p_start_of_human_token, p_end_of_human_token,
                              p_start_of_ai_token, p_end_of_ai_token,
                              p_start_of_speech_token, p_end_of_speech_token):
    # This is already batched, structure is good.
    global text_tokenizer
    if text_tokenizer is None: raise ValueError("text_tokenizer is not initialized.")

    input_ids_batch_list, labels_batch_list, attention_mask_batch_list = [], [], []
    num_examples_in_batch = len(examples_batch[list(examples_batch.keys())[0]])

    # Extract columns as lists for efficient iteration
    speaker_id_col = examples_batch.get('speaker_id', [None]*num_examples_in_batch)
    original_text_col = examples_batch.get('text', [None]*num_examples_in_batch)
    snac_codes_col = examples_batch.get('snac_codes', [None]*num_examples_in_batch)

    for i in range(num_examples_in_batch):
        speaker_id_val = speaker_id_col[i]
        original_text_val = original_text_col[i]
        snac_codes_val = snac_codes_col[i]

        if (speaker_id_val is None or
            original_text_val is None or not original_text_val or
            not isinstance(snac_codes_val, list) or not snac_codes_val):
            input_ids_batch_list.append(None)
            labels_batch_list.append(None)
            attention_mask_batch_list.append(None)
            continue
        try:
            # This part is per-sample, which is fine within a batched map function
            # The overhead is amortized over the batch.
            speaker_tag_prefix_for_this_sample = f"{speaker_id_val}: "
            text_to_encode_with_speaker_tag = speaker_tag_prefix_for_this_sample + original_text_val
            text_tokens_with_speaker = text_tokenizer.encode(text_to_encode_with_speaker_tag, add_special_tokens=False)

            input_ids_list_item = (
                [p_start_of_human_token] + text_tokens_with_speaker + [p_end_of_human_token] +
                [p_start_of_ai_token] + [p_start_of_speech_token] + snac_codes_val +
                [p_end_of_speech_token] + [p_end_of_ai_token]
            )
            input_ids_batch_list.append(input_ids_list_item)
            labels_batch_list.append(input_ids_list_item[:]) # Creates a copy
            attention_mask_batch_list.append([1] * len(input_ids_list_item))
        except Exception as e:
            input_ids_batch_list.append(None)
            labels_batch_list.append(None)
            attention_mask_batch_list.append(None)
    return {
        "input_ids": input_ids_batch_list,
        "labels": labels_batch_list,
        "attention_mask": attention_mask_batch_list
    }

def process_speaker_data_common_pipeline(dataset_for_processing, stage_name="speakers"):
    """Common processing pipeline for a dataset chunk."""
    global OPTIMIZED_NUM_PROC, OPTIMIZED_BATCH_SIZE # Use optimized globals

    print(f"\n--- Applying SNAC for {stage_name} ({len(dataset_for_processing)} samples) ---")
    # SNAC is GPU bound, num_proc=1 is usually best to avoid GPU contention.
    # Batching for map here doesn't help much as tokenize_audio_snac is per sample.
    processed_ds = dataset_for_processing.map(
        add_snac_codes_map_fn, num_proc=1, desc=f"SNAC for {stage_name}"
    )
    processed_ds = processed_ds.filter(lambda x: x["snac_codes"] is not None and len(x["snac_codes"]) > 0,
                                       num_proc=OPTIMIZED_NUM_PROC, desc=f"Filter empty SNAC for {stage_name}")
    if len(processed_ds) == 0:
        print(f"No data after SNAC for {stage_name}"); return None
    gc.collect()

    print(f"\n--- Deduplicating SNAC frames for {stage_name} ---")
    processed_ds = processed_ds.map(
        remove_duplicate_frames_batched_map_fn,
        batched=True, batch_size=OPTIMIZED_BATCH_SIZE, num_proc=OPTIMIZED_NUM_PROC,
        writer_batch_size=OPTIMIZED_BATCH_SIZE, # From suggestion
        desc=f"Deduplicating SNAC for {stage_name}"
    )
    processed_ds = processed_ds.rename_column("snac_codes", "snac_codes_original_pre_dedup")
    processed_ds = processed_ds.rename_column("snac_codes_deduped", "snac_codes")
    processed_ds = processed_ds.filter(
        lambda x: x["snac_codes"] is not None and len(x["snac_codes"]) > 0 and len(x["snac_codes"]) % 7 == 0,
        num_proc=OPTIMIZED_NUM_PROC, desc=f"Filter invalid deduped SNAC for {stage_name}"
    )
    if len(processed_ds) == 0:
        print(f"No data after dedup for {stage_name}"); return None
    gc.collect()

    print(f"\n--- Final formatting for {stage_name} ---")
    special_token_kwargs = {
        "p_start_of_human_token": start_of_human_token, "p_end_of_human_token": end_of_human_token,
        "p_start_of_ai_token": start_of_ai_token, "p_end_of_ai_token": end_of_ai_token,
        "p_start_of_speech_token": start_of_speech_token, "p_end_of_speech_token": end_of_speech_token,
    }
    # Determine columns to remove more robustly
    cols_to_remove_after_formatting = ["audio", "text", "snac_codes_original_pre_dedup", "snac_codes"]
    if 'audio_filename' in processed_ds.column_names: # If this column was added by local loading
        cols_to_remove_after_formatting.append('audio_filename')
    actual_cols_to_remove = [col for col in cols_to_remove_after_formatting if col in processed_ds.column_names]

    final_ds_chunk = processed_ds.map(
        create_final_input_batched_map_fn, fn_kwargs=special_token_kwargs,
        batched=True, batch_size=OPTIMIZED_BATCH_SIZE, num_proc=OPTIMIZED_NUM_PROC,
        remove_columns=actual_cols_to_remove,
        writer_batch_size=OPTIMIZED_BATCH_SIZE, # From suggestion
        desc=f"Final formatting for {stage_name}"
    )
    final_ds_chunk = final_ds_chunk.filter(lambda x: x["input_ids"] is not None,
                                           num_proc=OPTIMIZED_NUM_PROC, desc=f"Filter empty final inputs for {stage_name}")
    if len(final_ds_chunk) == 0:
        print(f"No data after final formatting for {stage_name}"); return None
    gc.collect()

    print(f"Casting final features for {stage_name}...")
    try:
        # Define the target features for the core training columns
        target_cast_features = {
            'input_ids': Sequence(Value(dtype='int64')),
            'labels': Sequence(Value(dtype='int64')),
            'attention_mask': Sequence(Value(dtype='int8'))
        }
        # Keep other existing columns (like speaker_id, sentence_id) as they are, if present
        current_features_dict = dict(final_ds_chunk.features)
        for col_name, feature_type in current_features_dict.items():
            if col_name not in target_cast_features:
                target_cast_features[col_name] = feature_type
        
        final_ds_chunk = final_ds_chunk.cast(Features(target_cast_features))
        if 'input_ids' in final_ds_chunk.features:
            print(f"Successfully cast features for {stage_name}. input_ids type: {final_ds_chunk.features['input_ids']}")
    except Exception as e:
        print(f"ERROR casting features for {stage_name}: {e}")
        import traceback; traceback.print_exc(); return None

    print(f"Finished processing {stage_name}. Output columns: {final_ds_chunk.column_names}, Count: {len(final_ds_chunk)}")
    return final_ds_chunk


def process_specified_speakers_from_hf_format(hf_dataset_repo_id_to_load, speakers_to_process_now):
    global OPTIMIZED_NUM_PROC # Use optimized global
    if not speakers_to_process_now:
        print("No new speakers specified for processing in process_specified_speakers_from_hf_format."); return None

    print(f"\n--- Loading Full Dataset from HF: {hf_dataset_repo_id_to_load} ---")
    try:
        dataset_full_raw = load_dataset(hf_dataset_repo_id_to_load, split="train", trust_remote_code=True)
        print(f"Loaded {len(dataset_full_raw)} raw samples from HF repo {hf_dataset_repo_id_to_load}")
        # Standardize column names early
        if 'hindi_sentence' in dataset_full_raw.column_names and 'text' not in dataset_full_raw.column_names:
            dataset_full_raw = dataset_full_raw.rename_column('hindi_sentence', 'text')
        elif 'sentence' in dataset_full_raw.column_names and 'text' not in dataset_full_raw.column_names:
             dataset_full_raw = dataset_full_raw.rename_column('sentence', 'text')

        if "speaker_id" not in dataset_full_raw.column_names:
            print(f"ERROR: 'speaker_id' column is MISSING from {hf_dataset_repo_id_to_load}. Found: {dataset_full_raw.column_names}"); return None
        required_cols = ['audio', 'speaker_id', 'text']
        if not all(col in dataset_full_raw.column_names for col in required_cols):
             print(f"ERROR: Dataset {hf_dataset_repo_id_to_load} missing essential columns {required_cols}. Found: {dataset_full_raw.column_names}"); return None

        if len(dataset_full_raw) > 0 and dataset_full_raw[0]['audio']['sampling_rate'] != TARGET_AUDIO_SAMPLING_RATE:
            print(f"Resampling audio to {TARGET_AUDIO_SAMPLING_RATE} Hz (this is a full dataset cast)...")
            dataset_full_raw = dataset_full_raw.cast_column("audio", Audio(sampling_rate=TARGET_AUDIO_SAMPLING_RATE))
            gc.collect()
    except Exception as e:
        print(f"Error loading raw dataset from {hf_dataset_repo_id_to_load}: {e}");
        import traceback; traceback.print_exc(); return None

    print(f"\n--- Filtering raw dataset for NEW speakers: {speakers_to_process_now} (using {OPTIMIZED_NUM_PROC} workers) ---")
    target_speaker_set = set(speakers_to_process_now)
    dataset_for_new_speakers = dataset_full_raw.filter(
        lambda ex: ex["speaker_id"] in target_speaker_set,
        num_proc=OPTIMIZED_NUM_PROC,
        desc="Filtering for new speakers"
    )
    if len(dataset_for_new_speakers) == 0:
        print(f"No data found for specified new speakers: {speakers_to_process_now} in dataset from {hf_dataset_repo_id_to_load}"); return None
    print(f"Selected {len(dataset_for_new_speakers)} samples for new speakers.")
    gc.collect()
    
    # Use the common processing pipeline
    return process_speaker_data_common_pipeline(dataset_for_new_speakers, stage_name="newly_loaded_speakers")


# --- Helper function to define Hub features ---
def get_hub_features_for_dataset(dataset: Dataset) -> Features:
    # ... (Your existing function is fine)
    hub_features_dict = {}
    standard_types = {
        'input_ids': Sequence(Value(dtype='int64')),
        'labels': Sequence(Value(dtype='int64')),
        'attention_mask': Sequence(Value(dtype='int8')),
        'speaker_id': Value('string'),
        'sentence_id': Value('string'),
    }
    for col_name in dataset.column_names:
        if col_name in standard_types:
            hub_features_dict[col_name] = standard_types[col_name]
        else:
            hub_features_dict[col_name] = dataset.features[col_name]
    return Features(hub_features_dict)

def push_dataset_to_hub(dataset_to_push: Dataset, hub_repo_id: str, is_private: bool, use_upload_folder: bool = False, temp_save_path: str = "./temp_dataset_for_upload"):
    """Pushes a dataset to the Hugging Face Hub with optimizations."""
    global OPTIMIZED_NUM_PROC, OPTIMIZED_MAX_SHARD_SIZE
    print(f"\n--- Pushing Dataset to Hub: {hub_repo_id} ---")
    try:
        final_hf_features = get_hub_features_for_dataset(dataset_to_push)
        # Ensure dataset columns match expected features before casting to avoid errors
        cols_to_keep_for_hub = [col for col in final_hf_features.keys() if col in dataset_to_push.column_names]
        dataset_aligned_for_hub = dataset_to_push.select_columns(cols_to_keep_for_hub)
        
        missing_hub_cols = [col for col in final_hf_features.keys() if col not in dataset_aligned_for_hub.column_names]
        if missing_hub_cols:
            print(f"WARNING: Dataset for Hub push is missing expected columns (will be absent in Hub): {missing_hub_cols}")
            # Adjust final_hf_features to only include present columns to avoid cast error
            final_hf_features = Features({k: v for k, v in final_hf_features.items() if k in dataset_aligned_for_hub.column_names})

        dataset_casted_for_push = dataset_aligned_for_hub.cast(final_hf_features)
        print(f"Dataset features cast for Hub push. Columns: {dataset_casted_for_push.column_names}")

        api = HfApi()
        api.create_repo(repo_id=hub_repo_id, repo_type="dataset", private=is_private, exist_ok=True)
        
        if use_upload_folder:
            print(f"Pre-saving dataset to {temp_save_path} for optimized upload...")
            if os.path.exists(temp_save_path):
                import shutil
                shutil.rmtree(temp_save_path) # Clean up previous attempt
            dataset_casted_for_push.save_to_disk(temp_save_path, num_proc=OPTIMIZED_NUM_PROC)
            print(f"Uploading from folder {temp_save_path} using {OPTIMIZED_NUM_PROC} threads...")
            upload_folder(
                repo_id=hub_repo_id,
                folder_path=temp_save_path,
                repo_type="dataset",
                # token=os.getenv("HF_TOKEN"), # Assumes token is set via login or env var
                path_in_repo="",
                multi_commits=True,
                multi_commits_verbose=True,
                threads=OPTIMIZED_NUM_PROC,
                commit_message=f"Upload dataset: {os.path.basename(hub_repo_id)}"
            )
            # Clean up temp folder
            # import shutil; shutil.rmtree(temp_save_path) # Optional: remove after upload
        else:
            print(f"Pushing {len(dataset_casted_for_push)} samples to {hub_repo_id} with max_shard_size={OPTIMIZED_MAX_SHARD_SIZE}...")
            dataset_casted_for_push.push_to_hub(
                hub_repo_id,
                private=is_private,
                max_shard_size=OPTIMIZED_MAX_SHARD_SIZE,
                num_proc=OPTIMIZED_NUM_PROC # For parallel Parquet writing
            )
        print(f"Dataset successfully pushed/uploaded to '{hub_repo_id}'")
    except Exception as e:
        print(f"Error pushing/uploading dataset to {hub_repo_id}: {e}"); import traceback; traceback.print_exc()


def run_append_speaker_workflow(existing_processed_dataset_id, new_speaker_ids_to_add, combined_dataset_hub_id, use_optimized_upload=False):
    global device, text_tokenizer, snac_model, DATASET_REPO_ID # DATASET_REPO_ID is where raw data comes from
    global OPTIMIZED_NUM_PROC, OPTIMIZED_BATCH_SIZE # Ensure these are accessible

    # Use the globally set device from main
    # device = "cuda" if torch.cuda.is_available() else "cpu" # Redundant if set in main
    load_models_and_tokenizer()

    print(f"\n--- Loading existing processed dataset: {existing_processed_dataset_id} ---")
    try:
        existing_dataset = load_dataset(existing_processed_dataset_id, split="train", trust_remote_code=True)
        print(f"Loaded {len(existing_dataset)} samples from {existing_processed_dataset_id}. Columns: {existing_dataset.column_names}")
    except Exception as e:
        print(f"Error loading existing dataset {existing_processed_dataset_id}: {e}"); exit(1)
    gc.collect()

    print(f"\n--- Processing new speaker(s): {new_speaker_ids_to_add} ---")
    processed_new_speaker_dataset = process_specified_speakers_from_hf_format(DATASET_REPO_ID, new_speaker_ids_to_add)

    if processed_new_speaker_dataset is None or len(processed_new_speaker_dataset) == 0:
        print("No data processed for new speakers. Aborting append workflow."); return
    gc.collect()

    # Optional: Save the newly processed speaker data separately
    new_speaker_suffix = "_".join(sorted([str(s).replace('/', '_').replace('\\', '_') for s in new_speaker_ids_to_add]))
    hub_repo_id_new_speakers_only = f"{os.path.dirname(combined_dataset_hub_id)}/{os.path.basename(combined_dataset_hub_id)}_new_speakers_{new_speaker_suffix}" # More robust naming
    print(f"\n--- Optionally pushing newly processed speaker data ({len(processed_new_speaker_dataset)} samples) to {hub_repo_id_new_speakers_only} ---")
    push_dataset_to_hub(processed_new_speaker_dataset, hub_repo_id_new_speakers_only, is_private=True, use_upload_folder=use_optimized_upload)
    gc.collect()

    print(f"\n--- Concatenating datasets ---")
    try:
        # Attempt to define common features based on the more rigorously processed new speaker dataset
        # Ensure both datasets only have columns intended for the final concatenated dataset before this.
        common_cols = list(set(existing_dataset.column_names).intersection(set(processed_new_speaker_dataset.column_names)))
        print(f"Common columns for concatenation: {common_cols}")

        essential_training_cols = {'input_ids', 'labels', 'attention_mask', 'speaker_id'}
        if not essential_training_cols.issubset(set(common_cols)):
            print(f"ERROR: Common columns {common_cols} do not contain all essential {essential_training_cols}.")
            return

        existing_dataset_aligned = existing_dataset.select_columns(common_cols)
        processed_new_speaker_dataset_aligned = processed_new_speaker_dataset.select_columns(common_cols)
        
        # Use features from the newly processed dataset as the canonical reference for casting
        # This helps ensure type consistency (e.g., int64 for input_ids)
        common_features_for_concat = get_hub_features_for_dataset(processed_new_speaker_dataset_aligned)
        print(f"Casting existing dataset to common features: {common_features_for_concat.keys()}")
        existing_dataset_casted = existing_dataset_aligned.cast(common_features_for_concat)
        
        # The new speaker dataset should already align, but casting ensures it.
        print(f"Casting new speaker dataset to common features (for safety): {common_features_for_concat.keys()}")
        processed_new_speaker_dataset_casted = processed_new_speaker_dataset_aligned.cast(common_features_for_concat)

        combined_dataset = concatenate_datasets([existing_dataset_casted, processed_new_speaker_dataset_casted])
    except Exception as e:
        print(f"Error during explicit casting or concatenation: {e}")
        import traceback; traceback.print_exc()
        print("Attempting concatenation with aligned columns only (less safe for types)...")
        try:
            combined_dataset = concatenate_datasets([existing_dataset.select_columns(common_cols),
                                                     processed_new_speaker_dataset.select_columns(common_cols)])
        except Exception as e_fallback:
            print(f"Fallback concatenation also failed: {e_fallback}"); return
    
    print(f"Combined dataset created with {len(combined_dataset)} samples. Columns: {combined_dataset.column_names}")
    gc.collect()

    push_dataset_to_hub(combined_dataset, combined_dataset_hub_id, is_private=True, use_upload_folder=use_optimized_upload)


def run_preprocessing_standalone(speakers_to_process_config=None, output_hub_repo_id=None, use_optimized_upload=False):
    global device, text_tokenizer, snac_model, DATASET_REPO_ID # Use global DATASET_REPO_ID
    global OPTIMIZED_NUM_PROC, OPTIMIZED_BATCH_SIZE # Use optimized globals

    start_time_prep = time.time()
    print(f"\n--- Initializing Standalone Preprocessing ---")
    # device is set in main
    load_models_and_tokenizer()

    print(f"\n--- Loading Dataset from HuggingFace: {DATASET_REPO_ID} ---")
    try:
        current_dataset = load_dataset(DATASET_REPO_ID, split="train", trust_remote_code=True)
        print(f"Loaded dataset with {len(current_dataset)} samples. Columns: {current_dataset.column_names}")
        # Standardize column names
        if 'hindi_sentence' in current_dataset.column_names and 'text' not in current_dataset.column_names:
             current_dataset = current_dataset.rename_column('hindi_sentence', 'text')
        elif 'sentence' in current_dataset.column_names and 'text' not in current_dataset.column_names:
             current_dataset = current_dataset.rename_column('sentence', 'text')
        required_cols = ['audio', 'speaker_id', 'text']
        if not all(col in current_dataset.column_names for col in required_cols):
             print(f"ERROR: Dataset {DATASET_REPO_ID} missing essential columns {required_cols}. Found: {current_dataset.column_names}"); exit(1)
    except Exception as e: print(f"ERROR loading dataset from {DATASET_REPO_ID}: {e}"); import traceback; traceback.print_exc(); exit(1)
    if len(current_dataset) == 0: print(f"ERROR: Loaded dataset {DATASET_REPO_ID} is empty."); exit(1)
    print(f"Dataset loaded in {time.time() - start_time_prep:.2f} seconds.")
    gc.collect()

    if speakers_to_process_config and isinstance(speakers_to_process_config, list) and len(speakers_to_process_config) > 0:
        start_time_filter = time.time()
        print(f"\n--- Filtering for speakers (using {OPTIMIZED_NUM_PROC} processes): {speakers_to_process_config} ---")
        initial_count = len(current_dataset)
        target_speaker_set = set(speakers_to_process_config)
        current_dataset = current_dataset.filter(
            lambda ex: ex["speaker_id"] in target_speaker_set,
            num_proc=OPTIMIZED_NUM_PROC,
            desc="Filtering speakers for standalone run"
        )
        print(f"Filtered out {initial_count - len(current_dataset)} samples in {time.time() - start_time_filter:.2f} seconds. Processing {len(current_dataset)}.")
        if len(current_dataset) == 0: print("ERROR: Dataset empty after speaker filter."); exit(1)
        gc.collect()
    else:
        print("\n--- Processing all speakers in loaded dataset for standalone run. ---")

    final_dataset = process_speaker_data_common_pipeline(current_dataset, stage_name="standalone_run")
    gc.collect()

    if final_dataset and len(final_dataset) > 0:
        if not output_hub_repo_id:
            print("ERROR: output_hub_repo_id must be provided for standalone preprocessing run.")
            exit(1)
        # Sample verification (optional, can be extensive)
        # ... (your existing sample verification logic can be called here) ...
        push_dataset_to_hub(final_dataset, output_hub_repo_id, is_private=True, use_upload_folder=use_optimized_upload)
    else:
        print("No data remaining to push after all processing steps in standalone run.")

    print("\n--- Standalone Preprocessing Script Finished ---")
    total_time = time.time() - start_time_prep
    print(f"Total processing time: {total_time:.2f} seconds ({total_time / 60:.2f} minutes)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process TTS dataset: append new speakers or run standalone.")
    parser.add_argument("--workflow", type=str, default="append", choices=["append", "standalone"],
                        help="Workflow to run: 'append' new speakers to existing, or 'standalone' full processing.")
    parser.add_argument("--dataset_id", type=str, default=DATASET_REPO_ID,
                       help=f"HuggingFace Hub ID of the RAW dataset to process (default: {DATASET_REPO_ID})")
    parser.add_argument("--existing_processed_dataset_id", type=str,
                        default="bharathkumar1922001/aisha-asmr-zadok-anika-ivanna-raju-sia-sangeeta",
                        help="[append workflow] HF Hub ID of existing PROCESSED dataset.")
    parser.add_argument("--new_speakers_to_add", nargs='+', default=[],
                        help="[append workflow] List of new speaker IDs to process and add (e.g., britanny rhea).")
    parser.add_argument("--combined_output_dataset_id", type=str,
                        default="bharathkumar1922001/aisha-asmr-zadok-anika-ivanna-raju-sia-sangeeta-brittany-rhea-shyam-shayana-knoxdork-nikita",
                        help="[append workflow] HF Hub ID for the final combined dataset.")
    parser.add_argument("--standalone_output_dataset_id", type=str, default=None,
                        help="[standalone workflow] HF Hub ID for the output of standalone processing. Required if workflow is standalone.")
    parser.add_argument("--speakers_for_standalone", nargs='+', default=None,
                        help="[standalone workflow] Specific speakers to process. If None, processes all from config or all in dataset.")
    parser.add_argument("--use_optimized_upload", action="store_true",
                        help="Use upload_folder for potentially faster Hub uploads (requires temp disk space).")
    parser.add_argument("--num_proc", type=int, default=OPTIMIZED_NUM_PROC,
                        help=f"Number of processes for CPU-bound ops (default: {OPTIMIZED_NUM_PROC} based on min(16, cores))")
    parser.add_argument("--batch_size_map", type=int, default=OPTIMIZED_BATCH_SIZE,
                        help=f"Batch size for .map() operations (default: {OPTIMIZED_BATCH_SIZE})")


    args = parser.parse_args()

    # Update globals based on args for performance tuning if provided
    OPTIMIZED_NUM_PROC = args.num_proc
    OPTIMIZED_BATCH_SIZE = args.batch_size_map
    print(f"Using OPTIMIZED_NUM_PROC: {OPTIMIZED_NUM_PROC}, OPTIMIZED_BATCH_SIZE: {OPTIMIZED_BATCH_SIZE}")


    # Set global DATASET_REPO_ID from args (this is the raw dataset source)
    DATASET_REPO_ID = args.dataset_id
    print(f"Using RAW dataset ID: {DATASET_REPO_ID}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Global device set to: {device} for main execution.")


    if args.workflow == "append":
        if not args.new_speakers_to_add:
            print("ERROR: For 'append' workflow, --new_speakers_to_add must be provided.")
            exit(1)
        run_append_speaker_workflow(
            existing_processed_dataset_id=args.existing_processed_dataset_id,
            new_speaker_ids_to_add=args.new_speakers_to_add,
            combined_dataset_hub_id=args.combined_output_dataset_id,
            use_optimized_upload=args.use_optimized_upload
        )
    elif args.workflow == "standalone":
        if not args.standalone_output_dataset_id:
            print("ERROR: For 'standalone' workflow, --standalone_output_dataset_id must be provided.")
            exit(1)
        
        speakers_for_run = args.speakers_for_standalone
        if not speakers_for_run: # If not provided via CLI, try config
            speakers_for_run = get_speakers_to_filter_from_config(CONFIG_FILE_PATH_FOR_PREPROCESS)
            if not speakers_for_run:
                print("No specific speakers for standalone run from CLI or config, will process all from dataset.")
        
        run_preprocessing_standalone(
            speakers_to_process_config=speakers_for_run,
            output_hub_repo_id=args.standalone_output_dataset_id,
            use_optimized_upload=args.use_optimized_upload
        )
    else:
        print(f"Unknown workflow: {args.workflow}")
        exit(1)