import os
import torch
import torchaudio.transforms as T
from datasets import load_dataset, Dataset, Features, Value, Audio, concatenate_datasets, Sequence # Added Sequence
from huggingface_hub import HfApi
from snac import SNAC
from transformers import AutoTokenizer
import random
import json
import numpy as np
from tqdm import tqdm
import yaml
import multiprocessing # Import multiprocessing to get cpu count
import time # For basic timing/debugging if needed
import gc
import argparse
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

# --- Configuration ---
CONFIG_FILE_PATH_FOR_PREPROCESS = "config.yaml"
# Updated to use dataset ID instead of local path
DATASET_REPO_ID = "bharathkumar1922001/hindi-speakers" # This is the main source of raw speaker data
# Local path for downloaded dataset (only used if not loading directly from HF)
LOCAL_RAW_DATASET_DIR = os.path.join(os.getcwd(), "downloaded_dataset_raw")
DOWNLOADED_DATASET_REPO_ID_FOR_INFO_ONLY = "bharathkumar1922001/hindi-speakers"
PROCESSED_HF_DATASET_REPO_FOR_UPLOAD = "bharathkumar1922001/hindi-speakers-processed" # Base name

ORPHEUS_TOKENIZER_NAME = "canopylabs/3b-hi-pretrain-research_release"
SNAC_MODEL_NAME = "hubertsiuzdak/snac_24khz"
TARGET_AUDIO_SAMPLING_RATE = 24000

# Define CPU worker configuration
CPU_MAP_NUM_WORKERS = max(1, multiprocessing.cpu_count() // 2) # Use half of available CPUs by default
CPU_MAP_BATCH_SIZE = 64  # Batch size for CPU processing operations

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
# num_workers = 1 # This was a local variable in run_preprocessing, not global.

def get_speakers_to_filter_from_config(config_path):
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

        if waveform.dim() == 2: waveform = waveform.unsqueeze(0) # SNAC expects [B, 1, T]

        with torch.inference_mode():
            codes = snac_model.encode(waveform) # codes is a list of tensors, [B, Q, T_codes]

        if not isinstance(codes, list) or len(codes) < 3: return None # Expecting 3 levels for this SNAC model

        codes_lvl0_cpu = codes[0].squeeze(1).cpu() # [B, T_codes] -> [T_codes] (assuming B=1)
        codes_lvl1_cpu = codes[1].squeeze(1).cpu() # [B, T_codes*2] -> [T_codes*2]
        codes_lvl2_cpu = codes[2].squeeze(1).cpu() # [B, T_codes*4] -> [T_codes*4]

        num_coarse_frames = codes_lvl0_cpu.shape[1]
        if num_coarse_frames == 0: return None

        # Check if finer levels have enough frames corresponding to coarse frames
        required_len_lvl1 = 2 * num_coarse_frames
        required_len_lvl2 = 4 * num_coarse_frames
        if codes_lvl1_cpu.shape[1] < required_len_lvl1 or codes_lvl2_cpu.shape[1] < required_len_lvl2:
            # This can happen for very short audios where resampling/padding inside SNAC might lead to discrepancies
            return None

        all_snac_tokens = []
        for i in range(num_coarse_frames):
            # Interleave tokens as per the 1, 2, 4, 4, 2, 4, 4 pattern
            all_snac_tokens.extend([
                codes_lvl0_cpu[0, i].item() + AUDIO_CODE_BASE_OFFSET,
                codes_lvl1_cpu[0, 2*i].item() + AUDIO_CODE_BASE_OFFSET + 4096,         # RVQ 2, frame 1
                codes_lvl2_cpu[0, 4*i].item() + AUDIO_CODE_BASE_OFFSET + (2*4096),     # RVQ 3, frame 1
                codes_lvl2_cpu[0, (4*i)+1].item() + AUDIO_CODE_BASE_OFFSET + (3*4096), # RVQ 4, frame 1
                codes_lvl1_cpu[0, (2*i)+1].item() + AUDIO_CODE_BASE_OFFSET + (4*4096), # RVQ 2, frame 2
                codes_lvl2_cpu[0, (4*i)+2].item() + AUDIO_CODE_BASE_OFFSET + (5*4096), # RVQ 3, frame 2
                codes_lvl2_cpu[0, (4*i)+3].item() + AUDIO_CODE_BASE_OFFSET + (6*4096)  # RVQ 4, frame 2
            ])
        return all_snac_tokens
    except Exception as e:
        # print(f"Error in tokenize_audio_snac: {e}") # Optional: for debugging individual errors
        return None


def add_snac_codes_map_fn(example):
    try:
        audio_data = example.get("audio")
        snac_tokens = tokenize_audio_snac(audio_data)
        example["snac_codes"] = snac_tokens
    except Exception as e:
        # print(f"Error processing audio for SNAC in map_fn: {e}")
        example["snac_codes"] = None
    return example

def remove_duplicate_frames_batched_map_fn(examples):
    results_batch = []
    for snac_tokens_list in examples['snac_codes']:
        if not isinstance(snac_tokens_list, list) or not snac_tokens_list or len(snac_tokens_list) % 7 != 0:
            results_batch.append(None) # Invalid input
            continue
        try:
            if len(snac_tokens_list) < 7: # Not even one full frame
                results_batch.append(snac_tokens_list if snac_tokens_list else None)
                continue

            result_tokens_single = snac_tokens_list[:7] # Start with the first frame
            for i in range(7, len(snac_tokens_list), 7):
                # Compare only the first token of the frame (level 0 token)
                if snac_tokens_list[i] != result_tokens_single[-7]:
                    result_tokens_single.extend(snac_tokens_list[i:i+7])
            results_batch.append(result_tokens_single if result_tokens_single else None)
        except Exception as e:
            # print(f"Error in remove_duplicate_frames_batched_map_fn: {e}")
            results_batch.append(None) # Error during processing
    return {"snac_codes_deduped": results_batch}

def create_final_input_batched_map_fn(examples,
                                      p_start_of_human_token, p_end_of_human_token,
                                      p_start_of_ai_token, p_end_of_ai_token,
                                      p_start_of_speech_token, p_end_of_speech_token):
    global text_tokenizer
    if text_tokenizer is None: raise ValueError("text_tokenizer is not initialized.")

    input_ids_batch, labels_batch, attention_mask_batch = [], [], []
    batch_size = len(examples[list(examples.keys())[0]]) # Get batch size from the first key

    for i in range(batch_size):
        speaker_id_val = examples.get('speaker_id', [None]*batch_size)[i]
        original_text = examples.get('text', [None]*batch_size)[i]
        snac_codes_val = examples.get('snac_codes', [None]*batch_size)[i] # This should be deduped codes

        if (speaker_id_val is None or
            original_text is None or not original_text or
            not isinstance(snac_codes_val, list) or not snac_codes_val):
            input_ids_batch.append(None)
            labels_batch.append(None)
            attention_mask_batch.append(None)
            continue

        try:
            speaker_tag_prefix_for_this_sample = f"{speaker_id_val}: "
            text_to_encode_with_speaker_tag = speaker_tag_prefix_for_this_sample + original_text
            text_tokens_with_speaker = text_tokenizer.encode(text_to_encode_with_speaker_tag, add_special_tokens=False)

            input_ids_list = (
                [p_start_of_human_token] + text_tokens_with_speaker + [p_end_of_human_token] +
                [p_start_of_ai_token] + [p_start_of_speech_token] + snac_codes_val +
                [p_end_of_speech_token] + [p_end_of_ai_token]
            )
            input_ids_batch.append(input_ids_list)
            labels_batch.append(input_ids_list[:]) # Labels are same as input_ids for auto-regressive model
            attention_mask_batch.append([1] * len(input_ids_list))
        except Exception as e:
            # print(f"Error in create_final_input_batched_map_fn for an item: {e}")
            input_ids_batch.append(None)
            labels_batch.append(None)
            attention_mask_batch.append(None)

    return {
        "input_ids": input_ids_batch,
        "labels": labels_batch,
        "attention_mask": attention_mask_batch
    }

def process_specified_speakers(organized_data_dir_path, speakers_to_process_now):
    # This function loads from a local JSON + WAV file structure.
    # It's kept for completeness but the HF path is preferred for the append workflow.
    global device, text_tokenizer, snac_model, CPU_MAP_NUM_WORKERS, CPU_MAP_BATCH_SIZE
    if not speakers_to_process_now:
        print("No new speakers specified for processing.")
        return None

    print(f"\n--- Loading Raw Data from Organized Directory: {organized_data_dir_path} ---")
    metadata_json_path = os.path.join(organized_data_dir_path, "organized_data.json")
    audio_files_base_dir = organized_data_dir_path # Assume audio files are relative to this dir

    if not os.path.exists(metadata_json_path):
        print(f"ERROR: Metadata file 'organized_data.json' not found in '{organized_data_dir_path}'.")
        return None

    try:
        with open(metadata_json_path, 'r', encoding='utf-8') as f:
            metadata_list_all_speakers = json.load(f)
        print(f"Loaded {len(metadata_list_all_speakers)} total metadata entries from '{metadata_json_path}'.")
        if not metadata_list_all_speakers:
            print("ERROR: Metadata list is empty."); return None
    except Exception as e:
        print(f"ERROR loading metadata from '{metadata_json_path}': {e}"); return None

    target_speaker_set = set(speakers_to_process_now)
    dataset_load_list_new_speakers = []
    missing_audio_files_count = 0
    print(f"\n--- Filtering metadata for NEW speakers: {speakers_to_process_now} ---")
    for item in metadata_list_all_speakers:
        if item.get("speaker_id") in target_speaker_set:
            # Assuming 'audio_filename' in metadata points to a file relative to audio_files_base_dir
            audio_path = os.path.join(audio_files_base_dir, item["audio_filename"])
            if not os.path.exists(audio_path):
                # print(f"Warning: Audio file not found: {audio_path}")
                missing_audio_files_count += 1
                continue
            dataset_load_list_new_speakers.append({
                "text": item["hindi_sentence"], # Or whatever your text field is named
                "speaker_id": item["speaker_id"],
                "sentence_id": item.get("sentence_id", ""), # Optional
                "audio_path_temp": audio_path # Temporary field to load audio
            })

    if missing_audio_files_count > 0:
        print(f"Warning: {missing_audio_files_count} audio files for new speakers not found and were skipped.")

    if not dataset_load_list_new_speakers:
        print(f"No data found (or audio files missing) for specified new speakers: {speakers_to_process_now}"); return None
    print(f"Prepared {len(dataset_load_list_new_speakers)} entries for new speakers to process.")

    # Create a Hugging Face Dataset from the list of dicts
    data_dict_new_speakers = {
        "text": [ex["text"] for ex in dataset_load_list_new_speakers],
        "speaker_id": [ex["speaker_id"] for ex in dataset_load_list_new_speakers],
        "sentence_id": [ex["sentence_id"] for ex in dataset_load_list_new_speakers],
        "audio_path_temp": [ex["audio_path_temp"] for ex in dataset_load_list_new_speakers]
    }
    interim_features = Features({
        'text': Value('string'), 'speaker_id': Value('string'),
        'sentence_id': Value('string'), 'audio_path_temp': Value('string')
    })
    dataset_for_new_speakers = Dataset.from_dict(data_dict_new_speakers, features=interim_features)

    # Load audio files into the 'audio' column
    def load_audio_and_cast_map_fn(example):
        example['audio'] = example['audio_path_temp'] # load_dataset will handle loading from path
        return example

    dataset_for_new_speakers = dataset_for_new_speakers.map(load_audio_and_cast_map_fn, remove_columns=['audio_path_temp'])
    dataset_for_new_speakers = dataset_for_new_speakers.cast_column('audio', Audio(sampling_rate=TARGET_AUDIO_SAMPLING_RATE))
    print(f"Initial dataset for new speakers created with {len(dataset_for_new_speakers)} samples.")

    print(f"\n--- Applying SNAC for new speakers ({len(dataset_for_new_speakers)} samples) ---")
    processed_new_speakers_ds = dataset_for_new_speakers.map(
        add_snac_codes_map_fn, num_proc=1, desc="SNAC for new speakers" # SNAC on GPU, so num_proc=1
    )
    processed_new_speakers_ds = processed_new_speakers_ds.filter(lambda x: x["snac_codes"] is not None and len(x["snac_codes"]) > 0)
    if len(processed_new_speakers_ds) == 0: print("No data after SNAC for new speakers"); return None

    print(f"\n--- Deduplicating for new speakers ---")
    processed_new_speakers_ds = processed_new_speakers_ds.map(
        remove_duplicate_frames_batched_map_fn, batched=True, batch_size=CPU_MAP_BATCH_SIZE, num_proc=CPU_MAP_NUM_WORKERS
    )
    processed_new_speakers_ds = processed_new_speakers_ds.rename_column("snac_codes", "snac_codes_original_pre_dedup")
    processed_new_speakers_ds = processed_new_speakers_ds.rename_column("snac_codes_deduped", "snac_codes")
    processed_new_speakers_ds = processed_new_speakers_ds.filter(
        lambda x: x["snac_codes"] is not None and len(x["snac_codes"]) > 0 and len(x["snac_codes"]) % 7 == 0
    )
    if len(processed_new_speakers_ds) == 0: print("No data after dedup for new speakers"); return None

    print(f"\n--- Final formatting for new speakers ---")
    special_token_kwargs = {
        "p_start_of_human_token": start_of_human_token, "p_end_of_human_token": end_of_human_token,
        "p_start_of_ai_token": start_of_ai_token, "p_end_of_ai_token": end_of_ai_token,
        "p_start_of_speech_token": start_of_speech_token, "p_end_of_speech_token": end_of_speech_token,
    }
    columns_to_remove_from_newly_processed = ["audio", "text", "snac_codes_original_pre_dedup", "snac_codes"]
    # If 'audio_filename' was part of the original dataset and carried through, remove it.
    # However, in this path, it's loaded as 'audio_path_temp' and removed earlier.
    # We just need to ensure the columns passed to create_final_input_batched_map_fn are correct.
    # The `remove_columns` in map will take care of what's left from `processed_new_speakers_ds`.
    actual_cols_to_remove = [col for col in columns_to_remove_from_newly_processed if col in processed_new_speakers_ds.column_names]

    final_new_speaker_ds = processed_new_speakers_ds.map(
        create_final_input_batched_map_fn, fn_kwargs=special_token_kwargs,
        batched=True, batch_size=CPU_MAP_BATCH_SIZE, num_proc=CPU_MAP_NUM_WORKERS,
        remove_columns=actual_cols_to_remove
    )
    final_new_speaker_ds = final_new_speaker_ds.filter(lambda x: x["input_ids"] is not None)
    if len(final_new_speaker_ds) == 0: print("No data after final formatting for new speakers"); return None
    
    print(f"Casting input_ids and labels to int64 for newly processed speakers (JSON/WAV path)...")
    try:
        target_features = {}
        if 'input_ids' in final_new_speaker_ds.column_names:
            target_features['input_ids'] = Sequence(Value(dtype='int64'))
        if 'labels' in final_new_speaker_ds.column_names:
            target_features['labels'] = Sequence(Value(dtype='int64'))
        if 'attention_mask' in final_new_speaker_ds.column_names: # Ensure attention_mask is also typed
            target_features['attention_mask'] = Sequence(Value(dtype='int8')) # Typically int8 for attention_mask

        # Preserve other columns like speaker_id, sentence_id
        for col_name, col_feature in final_new_speaker_ds.features.items():
            if col_name not in target_features:
                target_features[col_name] = col_feature # Keep existing type for other columns
        
        final_new_speaker_ds = final_new_speaker_ds.cast(Features(target_features))
        if 'input_ids' in final_new_speaker_ds.features:
            print(f"Successfully cast features for new speaker dataset. input_ids type: {final_new_speaker_ds.features['input_ids']}")
        else:
            print(f"Warning: 'input_ids' not found in features after casting for new speaker dataset.")

    except Exception as e:
        print(f"ERROR casting features for new speaker dataset (JSON/WAV path): {e}")
        import traceback; traceback.print_exc(); return None

    print(f"Finished processing new speakers. Output columns: {final_new_speaker_ds.column_names}")
    return final_new_speaker_ds

def process_specified_speakers_from_hf_format(hf_dataset_id_to_load, speakers_to_process_now):
    global device, text_tokenizer, snac_model, CPU_MAP_NUM_WORKERS, CPU_MAP_BATCH_SIZE
    if not speakers_to_process_now:
        print("No new speakers specified for processing in process_specified_speakers_from_hf_format."); return None

    print(f"\n--- Loading Full Dataset from HF: {hf_dataset_id_to_load} ---")
    try:
        dataset_full_raw = load_dataset(hf_dataset_id_to_load, split="train", trust_remote_code=True)
        print(f"Loaded {len(dataset_full_raw)} raw samples from HF repo {hf_dataset_id_to_load}")
        print(f"Columns: {dataset_full_raw.column_names}")

        # Standardize text column name
        if 'hindi_sentence' in dataset_full_raw.column_names and 'text' not in dataset_full_raw.column_names:
            dataset_full_raw = dataset_full_raw.rename_column('hindi_sentence', 'text')
        elif 'sentence' in dataset_full_raw.column_names and 'text' not in dataset_full_raw.column_names:
            dataset_full_raw = dataset_full_raw.rename_column('sentence', 'text')
        
        if "speaker_id" not in dataset_full_raw.column_names:
            print(f"ERROR: 'speaker_id' column is MISSING from the raw dataset loaded from {hf_dataset_id_to_load}.")
            print(f"Available columns: {dataset_full_raw.column_names}"); return None

        required_cols = ['audio', 'speaker_id', 'text'] # sentence_id is optional
        if not all(col in dataset_full_raw.column_names for col in required_cols):
            print(f"ERROR: Dataset from {hf_dataset_id_to_load} missing essential columns after rename (need {required_cols}). Found: {dataset_full_raw.column_names}"); return None

        # Ensure audio is at the target sampling rate
        # Check the sampling rate of the first audio item IF the dataset is not empty
        if len(dataset_full_raw) > 0 and isinstance(dataset_full_raw[0]['audio'], dict) and dataset_full_raw[0]['audio'].get('sampling_rate') != TARGET_AUDIO_SAMPLING_RATE:
            print(f"Resampling audio to {TARGET_AUDIO_SAMPLING_RATE} Hz...")
            dataset_full_raw = dataset_full_raw.cast_column("audio", Audio(sampling_rate=TARGET_AUDIO_SAMPLING_RATE))
        elif len(dataset_full_raw) > 0 and not isinstance(dataset_full_raw[0]['audio'], dict):
             print(f"Warning: Audio column in {hf_dataset_id_to_load} does not seem to be decoded Audio features. Skipping resampling check.")


    except Exception as e:
        print(f"Error loading raw dataset from HF format directory {hf_dataset_id_to_load}: {e}");
        import traceback; traceback.print_exc(); return None

    print(f"\n--- Filtering raw dataset for NEW speakers: {speakers_to_process_now} (using {CPU_MAP_NUM_WORKERS} workers) ---")
    target_speaker_set = set(speakers_to_process_now)
    dataset_for_new_speakers = dataset_full_raw.filter(
        lambda ex: ex["speaker_id"] in target_speaker_set,
        num_proc=CPU_MAP_NUM_WORKERS # Can use more workers for CPU-bound filtering
    )
    if len(dataset_for_new_speakers) == 0:
        print(f"No data found for specified new speakers: {speakers_to_process_now} in dataset from {hf_dataset_id_to_load}"); return None
    print(f"Processing {len(dataset_for_new_speakers)} samples for new speakers.")
    
    # Clean up large raw dataset if it's different from the filtered one
    if dataset_full_raw is not dataset_for_new_speakers:
        del dataset_full_raw
        gc.collect()


    print(f"\n--- Applying SNAC for new speakers ---")
    # SNAC model is on GPU, so num_proc=1 is generally best to avoid GPU contention
    # unless you have a multi-GPU setup and SNAC can leverage it or `map` distributes batches.
    # For a single GPU, `num_proc=1` is safest for the GPU-bound SNAC step.
    processed_ds = dataset_for_new_speakers.map(add_snac_codes_map_fn, num_proc=1, desc="SNAC for new speakers")
    processed_ds = processed_ds.filter(lambda x: x["snac_codes"] is not None and len(x["snac_codes"]) > 0)
    if len(processed_ds) == 0: print("No data after SNAC (HF format path)"); return None

    print(f"\n--- Deduplicating for new speakers ---")
    processed_ds = processed_ds.map(remove_duplicate_frames_batched_map_fn, batched=True, batch_size=CPU_MAP_BATCH_SIZE, num_proc=CPU_MAP_NUM_WORKERS)
    processed_ds = processed_ds.rename_column("snac_codes", "snac_codes_original_pre_dedup")
    processed_ds = processed_ds.rename_column("snac_codes_deduped", "snac_codes")
    processed_ds = processed_ds.filter(lambda x: x["snac_codes"] is not None and len(x["snac_codes"]) > 0 and len(x["snac_codes"]) % 7 == 0)
    if len(processed_ds) == 0: print("No data after dedup (HF format path)"); return None

    print(f"\n--- Final formatting for new speakers ---")
    special_token_kwargs = {
        "p_start_of_human_token": start_of_human_token, "p_end_of_human_token": end_of_human_token,
        "p_start_of_ai_token": start_of_ai_token, "p_end_of_ai_token": end_of_ai_token,
        "p_start_of_speech_token": start_of_speech_token, "p_end_of_speech_token": end_of_speech_token,
    }
    columns_to_remove = ["audio", "text", "snac_codes_original_pre_dedup", "snac_codes"]
    if 'audio_filename' in processed_ds.column_names: columns_to_remove.append('audio_filename') # If it exists
    actual_cols_to_remove = [col for col in columns_to_remove if col in processed_ds.column_names]

    final_new_speaker_ds = processed_ds.map(
        create_final_input_batched_map_fn, fn_kwargs=special_token_kwargs,
        batched=True, batch_size=CPU_MAP_BATCH_SIZE, num_proc=CPU_MAP_NUM_WORKERS,
        remove_columns=actual_cols_to_remove
    )
    final_new_speaker_ds = final_new_speaker_ds.filter(lambda x: x["input_ids"] is not None)
    if len(final_new_speaker_ds) == 0: print("No data after final format (HF format path)"); return None
    
    print(f"Casting input_ids and labels to int64 for newly processed speakers (HF format path)...")
    try:
        target_features_for_new_speaker_data = {}
        if 'input_ids' in final_new_speaker_ds.column_names:
            target_features_for_new_speaker_data['input_ids'] = Sequence(Value(dtype='int64'))
        if 'labels' in final_new_speaker_ds.column_names:
            target_features_for_new_speaker_data['labels'] = Sequence(Value(dtype='int64'))
        if 'attention_mask' in final_new_speaker_ds.column_names: # Ensure attention_mask is also typed
            target_features_for_new_speaker_data['attention_mask'] = Sequence(Value(dtype='int8'))
        
        # Preserve other columns like speaker_id, sentence_id
        for col_name, col_feature in final_new_speaker_ds.features.items():
            if col_name not in target_features_for_new_speaker_data:
                target_features_for_new_speaker_data[col_name] = col_feature
        
        final_new_speaker_ds = final_new_speaker_ds.cast(Features(target_features_for_new_speaker_data))
        if 'input_ids' in final_new_speaker_ds.features:
            print(f"Successfully cast features for new speaker dataset. input_ids type: {final_new_speaker_ds.features['input_ids']}")
        else:
            print(f"Warning: 'input_ids' not found in features after casting for new speaker dataset.")
    except Exception as e:
        print(f"ERROR casting features for new speaker dataset (HF format path): {e}")
        import traceback; traceback.print_exc(); return None
    
    print(f"Finished processing new speakers. Output columns: {final_new_speaker_ds.column_names}")
    return final_new_speaker_ds

# --- Helper function to define Hub features ---
def get_hub_features_for_dataset(dataset: Dataset) -> Features:
    hub_features_dict = {}
    # Define standard types for known columns
    standard_types = {
        'input_ids': Sequence(Value(dtype='int64')),
        'labels': Sequence(Value(dtype='int64')),
        'attention_mask': Sequence(Value(dtype='int8')),
        'speaker_id': Value('string'),
        'sentence_id': Value('string'), # If present and desired
    }
    for col_name in dataset.column_names:
        if col_name in standard_types:
            hub_features_dict[col_name] = standard_types[col_name]
        else:
            # For unknown columns, try to use their existing feature type
            # This handles cases where other columns might exist and have valid types
            if col_name in dataset.features:
                 hub_features_dict[col_name] = dataset.features[col_name]
                 print(f"Info: Column '{col_name}' in get_hub_features_for_dataset using existing feature: {dataset.features[col_name]}")
            else:
                # Fallback if feature not defined (should not happen for Dataset object)
                print(f"Warning: Column '{col_name}' in get_hub_features_for_dataset has no predefined feature. Defaulting to string.")
                hub_features_dict[col_name] = Value('string')
    
    # Ensure all *expected* standard columns are in the Features object,
    # even if not in the current dataset. This is more for creating a new dataset from dict.
    # For casting an existing dataset, this is less critical as it only casts present columns.
    # However, to be safe for push_to_hub, ensure the common features are there.
    for std_col, std_type in standard_types.items():
        if std_col not in hub_features_dict and std_col in dataset.column_names: # Only add if it was in dataset
             hub_features_dict[std_col] = std_type

    return Features(hub_features_dict)


def get_new_speakers_to_process(raw_data_repo_id: str, processed_data_repo_id: str) -> list:
    """
    Determines the list of speaker IDs present in the raw dataset 
    but not in the existing processed dataset.
    """
    print(f"\n--- Determining new speakers to process ---")
    print(f"Raw dataset repository: {raw_data_repo_id}")
    print(f"Processed dataset repository: {processed_data_repo_id}")

    all_raw_speakers = set()
    processed_speakers = set()

    # 1. Get all unique speaker IDs from the raw dataset
    try:
        print(f"Loading speaker IDs from raw dataset: {raw_data_repo_id}...")
        # Load only the 'speaker_id' column to be efficient if possible,
        # but load_dataset loads metadata first. Accessing the column directly is fine.
        raw_ds = load_dataset(raw_data_repo_id, split="train", trust_remote_code=True)
        if "speaker_id" not in raw_ds.column_names:
            print(f"ERROR: 'speaker_id' column not found in raw dataset {raw_data_repo_id}. Cannot determine new speakers.")
            return [] # Or raise an error
        all_raw_speakers = set(raw_ds["speaker_id"])
        print(f"Found {len(all_raw_speakers)} unique speakers in raw dataset: {sorted(list(all_raw_speakers))[:20]}...") # Print a sample
        del raw_ds
        gc.collect()
    except Exception as e:
        print(f"ERROR: Could not load or access speaker IDs from raw dataset {raw_data_repo_id}: {e}")
        import traceback; traceback.print_exc()
        return [] # Or raise an error

    # 2. Get all unique speaker IDs from the existing processed dataset
    if processed_data_repo_id:
        try:
            print(f"Loading speaker IDs from processed dataset: {processed_data_repo_id}...")
            processed_ds = load_dataset(processed_data_repo_id, split="train", trust_remote_code=True)
            if "speaker_id" not in processed_ds.column_names:
                print(f"Warning: 'speaker_id' column not found in processed dataset {processed_data_repo_id}. Assuming no speakers are processed or column name differs.")
                # If speaker_id is missing, we can't know who is processed, so we might reprocess or assume none.
                # For append, safer to assume none are there if column is missing, leading to reprocessing if data exists.
            else:
                processed_speakers = set(processed_ds["speaker_id"])
                print(f"Found {len(processed_speakers)} unique speakers in processed dataset: {sorted(list(processed_speakers))[:20]}...") # Print a sample
            del processed_ds
            gc.collect()
        except Exception as e:
            # This can happen if the processed_data_repo_id doesn't exist (e.g., first run)
            print(f"Info: Could not load processed dataset {processed_data_repo_id} (Reason: {e}). Assuming no speakers processed yet for this repo.")
            # processed_speakers remains an empty set
    else:
        print("Info: No existing processed dataset ID provided. Will process all speakers from raw dataset as new.")


    # 3. Calculate the difference
    new_speakers = list(all_raw_speakers - processed_speakers)
    print(f"Speakers in raw dataset but not in processed: {len(new_speakers)}")
    if new_speakers:
        print(f"Example new speakers to process: {sorted(new_speakers)[:20]}...")
    else:
        print("No new speakers found to process. All speakers from raw dataset are already in the processed dataset (or raw dataset is empty/inaccessible).")

    return sorted(new_speakers)


def run_append_speaker_workflow(existing_processed_dataset_id, new_speaker_ids_to_add, combined_dataset_hub_id):
    global device, text_tokenizer, snac_model, CPU_MAP_NUM_WORKERS, CPU_MAP_BATCH_SIZE, start_of_human_token, end_of_human_token, start_of_ai_token, end_of_ai_token, start_of_speech_token, end_of_speech_token, AUDIO_CODE_BASE_OFFSET, TARGET_AUDIO_SAMPLING_RATE, ORPHEUS_TOKENIZER_NAME, SNAC_MODEL_NAME, DATASET_REPO_ID

    # Ensure CPU_MAP_NUM_WORKERS is set for this workflow (e.g., full CPUs if desired)
    # CPU_MAP_NUM_WORKERS = max(1, multiprocessing.cpu_count()) # Example: use all CPUs
    # print(f"Using {CPU_MAP_NUM_WORKERS} CPU workers for append workflow map operations.")


    device = "cuda" if torch.cuda.is_available() else "cpu"
    load_models_and_tokenizer()

    existing_dataset = None
    if existing_processed_dataset_id:
        print(f"\n--- Loading existing processed dataset: {existing_processed_dataset_id} ---")
        try:
            existing_dataset = load_dataset(existing_processed_dataset_id, split="train", trust_remote_code=True)
            print(f"Loaded {len(existing_dataset)} samples from {existing_processed_dataset_id}. Columns: {existing_dataset.column_names}")
            # Basic validation
            required_cols_existing = {'input_ids', 'labels', 'attention_mask', 'speaker_id'}
            if not required_cols_existing.issubset(set(existing_dataset.column_names)):
                print(f"WARNING: Existing dataset {existing_processed_dataset_id} is missing one or more required columns ({required_cols_existing}). This might cause issues.")
        except Exception as e:
            print(f"Warning: Error loading existing dataset {existing_processed_dataset_id}: {e}. Will proceed as if it's empty or non-existent for concatenation.")
            existing_dataset = None # Treat as if no existing dataset
    else:
        print("No existing_processed_dataset_id provided. Will create dataset only from new speakers.")


    if not new_speaker_ids_to_add:
        print("No new speaker IDs to process. ")
        if existing_dataset:
            print("If you only wanted to re-push the existing dataset, this script doesn't do that directly.")
            print("To re-push, you might load it and push to combined_dataset_hub_id manually or adjust logic.")
            # Optional: If the goal is to push the existing_dataset to combined_dataset_hub_id if no new speakers
            # This is not currently the logic, but could be added if `combined_dataset_hub_id` is different
            # and the user wants to "copy" the existing to the new combined ID.
        else:
            print("And no existing dataset was loaded. Nothing to do for append workflow.")
        return

    print(f"\n--- Processing new speaker(s): {new_speaker_ids_to_add} ---")
    # Load raw data for NEW speakers directly from the main HuggingFace repository
    print(f"Loading raw data for new speakers from HuggingFace repository: {DATASET_REPO_ID}") # DATASET_REPO_ID is global for raw data
    processed_new_speaker_dataset = process_specified_speakers_from_hf_format(DATASET_REPO_ID, new_speaker_ids_to_add)

    if processed_new_speaker_dataset is None or len(processed_new_speaker_dataset) == 0:
        print("No data processed for new speakers.")
        if existing_dataset is None:
            print("And no existing dataset loaded. Aborting append workflow as there's no data to push."); return
        else:
            print("Will proceed to push only the existing dataset to the combined_dataset_hub_id.")
            # Fall through to push existing_dataset if it exists

    # --- Save the newly processed speaker data (Your requested change) ---
    if processed_new_speaker_dataset and len(processed_new_speaker_dataset) > 0:
        print(f"\n--- Saving newly processed speaker data ({len(processed_new_speaker_dataset)} samples) ---")
        new_speaker_suffix = "_".join(sorted([str(s).replace('/', '_').replace('\\', '_') for s in new_speaker_ids_to_add]))
        if len(new_speaker_suffix) > 60: # Truncate if too long
            new_speaker_suffix = f"{len(new_speaker_ids_to_add)}new_speakers_hash{hash(new_speaker_suffix) % 10000}"

        # 1. Local save
        local_save_path_new_speakers = f"./processed_data_new_speakers_only_{new_speaker_suffix}"
        if not os.path.exists(local_save_path_new_speakers): # save_to_disk expects a directory name
             os.makedirs(local_save_path_new_speakers, exist_ok=True)
        
        try:
            print(f"Saving newly processed speaker data locally to directory: {local_save_path_new_speakers}")
            processed_new_speaker_dataset.save_to_disk(local_save_path_new_speakers)
            print(f"Successfully saved newly processed speaker data locally to '{local_save_path_new_speakers}'.")
        except Exception as e:
            print(f"Error saving newly processed speaker data locally: {e}")
            # Decide if this is fatal. For now, we'll try to proceed with Hub push if local save fails.

        # 2. Hub push (optional, but good for backup of this intermediate step)
        # Use a distinct name for this intermediate dataset on the Hub.
        hub_repo_id_new_speakers_only = f"{combined_dataset_hub_id}_temp_new_speakers_{new_speaker_suffix}"
        try:
            print(f"Pushing newly processed speaker data (only new speakers) to Hub: {hub_repo_id_new_speakers_only}")
            api = HfApi()
            # Ensure repo name is valid for HF
            hub_repo_id_new_speakers_only = hub_repo_id_new_speakers_only.replace(" ", "-") # Basic sanitization
            api.create_repo(repo_id=hub_repo_id_new_speakers_only, repo_type="dataset", private=True, exist_ok=True)
            
            hub_features_for_new_speakers = get_hub_features_for_dataset(processed_new_speaker_dataset)
            dataset_for_new_speaker_push = processed_new_speaker_dataset.cast(hub_features_for_new_speakers)

            dataset_for_new_speaker_push.push_to_hub(hub_repo_id_new_speakers_only, private=True)
            print(f"Successfully pushed newly processed speaker data to '{hub_repo_id_new_speakers_only}'")
        except Exception as e:
            print(f"Error pushing newly processed speaker data (only new speakers) to Hub: {e}")
            import traceback; traceback.print_exc()
            # Decide if this is fatal. If local save succeeded, you might still proceed with concatenation.
    # --- End of saving newly processed speaker data ---

    print(f"\n--- Preparing for final dataset push to {combined_dataset_hub_id} ---")
    
    datasets_to_combine = []
    if existing_dataset and len(existing_dataset) > 0:
        datasets_to_combine.append(existing_dataset)
    if processed_new_speaker_dataset and len(processed_new_speaker_dataset) > 0:
        datasets_to_combine.append(processed_new_speaker_dataset)

    if not datasets_to_combine:
        print("No data available (neither existing nor newly processed) to push. Exiting.")
        return

    combined_dataset = None
    if len(datasets_to_combine) == 1:
        print("Only one dataset part available (either existing or new). Using it as the combined dataset.")
        combined_dataset = datasets_to_combine[0]
    else: # len == 2
        print(f"Concatenating existing ({len(datasets_to_combine[0])} samples) and new ({len(datasets_to_combine[1])} samples) datasets.")
        # Align columns and features before concatenation
        common_cols = list(set(datasets_to_combine[0].column_names).intersection(set(datasets_to_combine[1].column_names)))
        print(f"Common columns for concatenation: {common_cols}")

        essential_training_cols = {'input_ids', 'labels', 'attention_mask', 'speaker_id'}
        if not essential_training_cols.issubset(set(common_cols)):
            print(f"ERROR: Common columns {common_cols} do not contain all essential training columns {essential_training_cols}.")
            print(f"  Existing dataset columns: {datasets_to_combine[0].column_names}")
            print(f"  New speaker dataset columns: {datasets_to_combine[1].column_names}")
            print("Cannot safely concatenate. Please ensure both datasets have the essential columns with consistent naming and types.")
            return

        existing_ds_aligned = datasets_to_combine[0].select_columns(common_cols)
        new_ds_aligned = datasets_to_combine[1].select_columns(common_cols)

        try:
            # Use features from the newly processed dataset as the reference for casting,
            # as it should be correctly typed by the processing pipeline.
            # However, it's better to define common_features based on 'get_hub_features_for_dataset'
            # applied to one of the aligned datasets.
            ref_dataset_for_features = new_ds_aligned # Or existing_ds_aligned if preferred
            common_features_for_concat = get_hub_features_for_dataset(ref_dataset_for_features.select_columns(common_cols))
            
            print(f"Casting both datasets to common features for concatenation: {common_features_for_concat.keys()}")
            
            existing_ds_casted = existing_ds_aligned.cast(common_features_for_concat)
            new_ds_casted = new_ds_aligned.cast(common_features_for_concat)
            
            combined_dataset = concatenate_datasets([existing_ds_casted, new_ds_casted])
        except Exception as e:
            print(f"Error during explicit casting or concatenation: {e}")
            print("Attempting concatenation with aligned columns but without explicit pre-casting (less safe)...")
            import traceback; traceback.print_exc()
            try:
                combined_dataset = concatenate_datasets([existing_ds_aligned, new_ds_aligned])
            except Exception as e_concat_fallback:
                print(f"Fallback concatenation also failed: {e_concat_fallback}")
                print("Aborting push. Please check dataset compatibility.")
                return
    
    if combined_dataset is None or len(combined_dataset) == 0:
        print("Combined dataset is empty. Nothing to push.")
        return

    print(f"Combined dataset created with {len(combined_dataset)} samples. Columns: {combined_dataset.column_names}")

    print(f"\n--- Pushing Combined Dataset to Hub: {combined_dataset_hub_id} ---")
    try:
        final_hf_features_for_combined = get_hub_features_for_dataset(combined_dataset)
        # Ensure all columns in combined_dataset are covered by final_hf_features_for_combined
        # And that essential columns are correctly typed (int64 for IDs, int8 for mask)
        print(f"Casting combined dataset to final features: {final_hf_features_for_combined.keys()}")
        dataset_for_push = combined_dataset.cast(final_hf_features_for_combined)
        
        api = HfApi()
        api.create_repo(repo_id=combined_dataset_hub_id, repo_type="dataset", private=True, exist_ok=True) # Make private=False for public
        dataset_for_push.push_to_hub(combined_dataset_hub_id, private=True) # Make private=False for public
        print(f"Successfully pushed combined dataset to '{combined_dataset_hub_id}'")
    except Exception as e:
        print(f"Error pushing combined dataset: {e}"); import traceback; traceback.print_exc()


def run_preprocessing(speakers_to_process=None):
    global device, text_tokenizer, snac_model, PROCESSED_HF_DATASET_REPO_FOR_UPLOAD, CPU_MAP_NUM_WORKERS, CPU_MAP_BATCH_SIZE

    start_time_prep = time.time()
    print(f"\n--- Initializing Preprocessing ---")
    # CPU_MAP_NUM_WORKERS and CPU_MAP_BATCH_SIZE are global, ensure they are set as desired for this path
    num_workers_local = CPU_MAP_NUM_WORKERS # Use a local var for clarity if preferred
    batch_size_cpu_local = CPU_MAP_BATCH_SIZE

    device = "cuda" if torch.cuda.is_available() else "cpu" # Ensure device is set for this run
    print(f"Using configuration:")
    print(f"  - Target device for SNAC: {device}")
    print(f"  - SNAC Map Processes: 1") # SNAC usually on GPU, 1 proc
    print(f"  - CPU Map Processes (filter, dedup, format): {num_workers_local}")
    print(f"  - CPU Map Batch Size: {batch_size_cpu_local}")

    load_models_and_tokenizer()

    print(f"\n--- Loading Dataset from HuggingFace: {DATASET_REPO_ID} ---")
    try:
        current_dataset = load_dataset(DATASET_REPO_ID, split="train", trust_remote_code=True)
        print(f"Loaded dataset with {len(current_dataset)} samples. Columns: {current_dataset.column_names}")
        if 'hindi_sentence' in current_dataset.column_names and 'text' not in current_dataset.column_names:
            current_dataset = current_dataset.rename_column('hindi_sentence', 'text')
        elif 'sentence' in current_dataset.column_names and 'text' not in current_dataset.column_names:
            current_dataset = current_dataset.rename_column('sentence', 'text')

        required_cols = ['audio', 'speaker_id', 'text']
        if not all(col in current_dataset.column_names for col in required_cols):
            print(f"ERROR: Dataset missing essential columns ({required_cols}). Found: {current_dataset.column_names}"); exit(1)
        
        if len(current_dataset) > 0 and isinstance(current_dataset[0]['audio'], dict) and current_dataset[0]['audio'].get('sampling_rate') != TARGET_AUDIO_SAMPLING_RATE:
            print(f"Resampling audio to {TARGET_AUDIO_SAMPLING_RATE} Hz for run_preprocessing path...")
            current_dataset = current_dataset.cast_column("audio", Audio(sampling_rate=TARGET_AUDIO_SAMPLING_RATE))

    except Exception as e: print(f"ERROR loading dataset: {e}"); import traceback; traceback.print_exc(); exit(1)
    if len(current_dataset) == 0: print("ERROR: Loaded dataset is empty."); exit(1)
    print(f"Dataset loaded in {time.time() - start_time_prep:.2f} seconds.")
    gc.collect()

    if speakers_to_process and isinstance(speakers_to_process, list) and len(speakers_to_process) > 0:
        start_time_filter = time.time()
        print(f"\n--- Filtering for speakers (using {num_workers_local} processes): {speakers_to_process} ---")
        initial_count = len(current_dataset)
        target_speaker_set = set(speakers_to_process)
        current_dataset = current_dataset.filter(
            lambda ex: ex["speaker_id"] in target_speaker_set,
            num_proc=num_workers_local # CPU bound filter
        )
        print(f"Filtered out {initial_count - len(current_dataset)} samples in {time.time() - start_time_filter:.2f} seconds. Processing {len(current_dataset)}.")
        if len(current_dataset) == 0: print("ERROR: Dataset empty after speaker filter."); exit(1)
        gc.collect()
    else:
        print("\n--- Processing all speakers in loaded dataset. ---")

    start_time_snac = time.time()
    print(f"\n--- Applying SNAC Audio Tokenization (using 1 process, device: {device}) ---")
    current_dataset = current_dataset.map(
        add_snac_codes_map_fn, num_proc=1, desc="Tokenizing Audio with SNAC" # GPU bound
    )
    count_before_filter = len(current_dataset)
    current_dataset = current_dataset.filter(lambda x: x["snac_codes"] is not None and len(x["snac_codes"]) > 0)
    print(f"Filtered {count_before_filter - len(current_dataset)} samples with invalid/empty SNAC codes.")
    print(f"SNAC Tokenization and filtering finished in {time.time() - start_time_snac:.2f} seconds. Size after SNAC: {len(current_dataset)}")
    if len(current_dataset) == 0: print("CRITICAL: No data remaining after SNAC processing and filtering."); exit(1)
    gc.collect()

    start_time_dedup = time.time()
    print(f"\n--- Removing Duplicate Audio Frames (using {num_workers_local} processes, batched, batch_size={batch_size_cpu_local}) ---")
    current_dataset = current_dataset.map(
        remove_duplicate_frames_batched_map_fn,
        batched=True, batch_size=batch_size_cpu_local, num_proc=num_workers_local, # CPU bound
        desc="Deduplicating audio frames"
    )
    current_dataset = current_dataset.rename_column("snac_codes", "snac_codes_original_pre_dedup")
    current_dataset = current_dataset.rename_column("snac_codes_deduped", "snac_codes")
    count_before_filter = len(current_dataset)
    current_dataset = current_dataset.filter(
        lambda x: x["snac_codes"] is not None and len(x["snac_codes"]) > 0 and len(x["snac_codes"]) % 7 == 0
    )
    print(f"Filtered {count_before_filter - len(current_dataset)} samples after deduplication filtering.")
    print(f"Deduplication and filtering finished in {time.time() - start_time_dedup:.2f} seconds. Size after Dedup: {len(current_dataset)}")
    if len(current_dataset) == 0: print("CRITICAL: No data remaining after deduplication and filtering."); exit(1)
    gc.collect()

    start_time_format = time.time()
    print(f"\n--- Creating Final Input IDs (using {num_workers_local} processes, batched, batch_size={batch_size_cpu_local}) ---")
    columns_to_remove_final = ["audio", "text", "snac_codes_original_pre_dedup", "snac_codes"]
    if 'audio_filename' in current_dataset.column_names: columns_to_remove_final.append('audio_filename')
    actual_cols_to_remove = [col for col in columns_to_remove_final if col in current_dataset.column_names]
    print(f"Columns to be removed: {actual_cols_to_remove}")
    special_token_kwargs_for_map = {
        "p_start_of_human_token": start_of_human_token, "p_end_of_human_token": end_of_human_token,
        "p_start_of_ai_token": start_of_ai_token, "p_end_of_ai_token": end_of_ai_token,
        "p_start_of_speech_token": start_of_speech_token, "p_end_of_speech_token": end_of_speech_token,
    }
    final_dataset = current_dataset.map(
        create_final_input_batched_map_fn,
        fn_kwargs=special_token_kwargs_for_map,
        batched=True, batch_size=batch_size_cpu_local, num_proc=num_workers_local, # CPU bound
        remove_columns=actual_cols_to_remove,
        desc="Creating Final Input Sequences"
    )
    count_before_filter = len(final_dataset)
    final_dataset = final_dataset.filter(lambda x: x["input_ids"] is not None)
    print(f"Filtered {count_before_filter - len(final_dataset)} samples that failed final processing.")
    print(f"Final formatting and filtering finished in {time.time() - start_time_format:.2f} seconds. Final Size: {len(final_dataset)}")
    if len(final_dataset) == 0: print("CRITICAL: No data remaining after final processing and filtering."); exit(1)
    print(f"Final dataset columns: {final_dataset.column_names}")
    gc.collect()

    print(f"\n--- Detailed View of {min(3, len(final_dataset))} Random Final Processed Samples ---")
    num_samples_to_show = min(3, len(final_dataset))
    if num_samples_to_show > 0 and text_tokenizer is not None:
        random_indices = random.sample(range(len(final_dataset)), num_samples_to_show)
        for i, original_idx in enumerate(random_indices):
            try:
                sample = final_dataset[original_idx]
                print(f"\n--- Sample {i} (Index in final_dataset: {original_idx}) ---")
                input_ids = sample.get("input_ids", [])
                speaker_id_check = sample.get("speaker_id", "N/A_col_absent")
                sentence_id_check = sample.get("sentence_id", "N/A_col_absent")
                print(f"  Speaker ID (from data field): {speaker_id_check}")
                if sentence_id_check != "N/A_col_absent": print(f"  Sentence ID (from data field): {sentence_id_check}")
                print(f"  Length of input_ids: {len(input_ids)}")

                if not input_ids: print("  Input IDs list is empty!"); continue

                soh_idx = input_ids.index(start_of_human_token) if start_of_human_token in input_ids else -1
                eoh_idx = input_ids.index(end_of_human_token) if end_of_human_token in input_ids else -1
                if soh_idx != -1 and eoh_idx != -1 and soh_idx < eoh_idx:
                    text_toks = input_ids[soh_idx+1:eoh_idx]
                    decoded_text = text_tokenizer.decode(text_toks, skip_special_tokens=True)
                    print(f"  Decoded Text (SOH to EOH): '{decoded_text}'")
                    if speaker_id_check != "N/A_col_absent" and not decoded_text.startswith(speaker_id_check + ": "):
                        print(f"    VERIFICATION WARNING: Decoded text prefix mismatch. Expected '{speaker_id_check}: '. Found: '{decoded_text[:len(speaker_id_check)+10]}...'")
                else:
                    print("  Could not find SOH/EOH tokens to decode text segment.")

                soa_idx = input_ids.index(start_of_ai_token) if start_of_ai_token in input_ids else -1
                sos_idx = input_ids.index(start_of_speech_token) if start_of_speech_token in input_ids else -1
                eos_idx = input_ids.index(end_of_speech_token) if end_of_speech_token in input_ids else -1
                eoa_idx = input_ids.index(end_of_ai_token) if end_of_ai_token in input_ids else -1

                # Verify structure
                # SOH ... EOH SOA SOS ... EOS EOA
                structure_ok = (soh_idx == 0 and
                                eoh_idx > soh_idx and
                                soa_idx == eoh_idx + 1 and
                                sos_idx == soa_idx + 1 and
                                eos_idx > sos_idx and
                                eoa_idx == eos_idx + 1 and
                                eoa_idx == len(input_ids) - 1)
                if structure_ok:
                    print(f"  Structure Verified: SOH...EOH SOA SOS ... EOS EOA")
                else:
                    print(f"    STRUCTURE WARNING: Special token sequence seems incorrect.")
                    print(f"      Indices: SOH={soh_idx}, EOH={eoh_idx}, SOA={soa_idx}, SOS={sos_idx}, EOS={eos_idx}, EOA={eoa_idx}, len={len(input_ids)}")

            except Exception as e:
                print(f"  Error during sample verification display for index {original_idx}: {e}")

    if len(final_dataset) > 0:
        # PROCESSED_HF_DATASET_REPO_FOR_UPLOAD is global
        print(f"\n--- Pushing Processed Dataset to Hugging Face Hub: {PROCESSED_HF_DATASET_REPO_FOR_UPLOAD} ---")
        try:
            hub_features = get_hub_features_for_dataset(final_dataset)
            print(f"Defining Hub features based on final columns: {list(hub_features.keys())}")

            # Align dataset columns to hub_features before casting (removes extra, checks missing)
            cols_to_keep = [col for col in hub_features.keys() if col in final_dataset.column_names]
            if set(cols_to_keep) != set(final_dataset.column_names):
                cols_to_remove_before_cast = list(set(final_dataset.column_names) - set(cols_to_keep))
                print(f"Removing columns not in Hub features before casting: {cols_to_remove_before_cast}")
                final_dataset_aligned = final_dataset.remove_columns(cols_to_remove_before_cast)
            else:
                final_dataset_aligned = final_dataset
            
            missing_cols = [col for col in hub_features.keys() if col not in final_dataset_aligned.column_names and col in ['input_ids', 'labels', 'attention_mask', 'speaker_id']] # Check essential
            if missing_cols:
                print(f"ERROR: Final dataset is missing required Hub feature columns after alignment: {missing_cols}")
                exit(1)

            print("Casting dataset features for Hub compatibility...")
            dataset_for_push = final_dataset_aligned.cast(hub_features)

            api = HfApi()
            api.create_repo(repo_id=PROCESSED_HF_DATASET_REPO_FOR_UPLOAD, repo_type="dataset", private=True, exist_ok=True) # Make private=False for public
            print(f"Pushing {len(dataset_for_push)} samples to {PROCESSED_HF_DATASET_REPO_FOR_UPLOAD}...")
            dataset_for_push.push_to_hub(PROCESSED_HF_DATASET_REPO_FOR_UPLOAD, private=True) # Make private=False for public
            print(f"Processed dataset successfully pushed to '{PROCESSED_HF_DATASET_REPO_FOR_UPLOAD}'")
        except Exception as e:
            print(f"Error pushing dataset: {e}"); import traceback; traceback.print_exc()
    else:
        print("No data remaining to push after all processing steps.")

    print("\n--- Preprocessing Script Finished ---")
    total_time = time.time() - start_time_prep
    print(f"Total processing time: {total_time:.2f} seconds ({total_time / 60:.2f} minutes)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process the TTS dataset")
    parser.add_argument("--raw_dataset_id", type=str, default=DATASET_REPO_ID,
                        help=f"HuggingFace dataset ID for raw audio data (default: {DATASET_REPO_ID})")
    parser.add_argument("--existing_processed_dataset_id", type=str,
                        default="bharathkumar1922001/aisha-asmr-zadok-anika-ivanna-raju-sia-sangeeta-brittany-rhea-shyam-shayana-knoxdork-nikita", # Example, adjust as needed
                        help="HuggingFace dataset ID of existing processed dataset to append to. Leave empty or omit if starting fresh/no append.")
    parser.add_argument("--output_dataset_id", type=str,
                        default="bharathkumar1922001/processed-hindi-data-combined", # Example, adjust as needed
                        help="HuggingFace dataset ID for the output (combined or new) processed dataset.")
    parser.add_argument("--mode", type=str, choices=['append', 'preprocess_config'], default='append',
                        help="Mode of operation: 'append' to add new speakers to an existing dataset, "
                             "'preprocess_config' to process speakers based on config.yaml.")
    parser.add_argument("--cpu_workers", type=int, default=None,
                        help="Number of CPU workers for parallel processing. Default uses half of CPU cores for preprocess_config, all cores for append.")


    args = parser.parse_args()

    # Update global DATASET_REPO_ID if provided (source of raw data)
    if args.raw_dataset_id != DATASET_REPO_ID:
        DATASET_REPO_ID = args.raw_dataset_id
        print(f"Using RAW dataset ID from command line: {DATASET_REPO_ID}")
    
    # Set CPU worker configuration based on mode or argument
    if args.cpu_workers is not None:
        CPU_MAP_NUM_WORKERS = max(1, args.cpu_workers)
    elif args.mode == 'append':
        CPU_MAP_NUM_WORKERS = max(1, multiprocessing.cpu_count()) # Default to all CPUs for append
    else: # preprocess_config or default
        CPU_MAP_NUM_WORKERS = max(1, multiprocessing.cpu_count()) # Default to half CPUs

    print(f"Setting CPU_MAP_NUM_WORKERS to {CPU_MAP_NUM_WORKERS} for this run.")


    device = "cuda" if torch.cuda.is_available() else "cpu" # Ensure device is set globally
    print(f"Global device set to: {device} for main execution.")


    if args.mode == 'append':
        print("\n--- Running in APPEND mode ---")
        EXISTING_PROCESSED_DATASET_ID_ON_HUB = args.existing_processed_dataset_id
        COMBINED_DATASET_NEW_HUB_ID = args.output_dataset_id

        # Dynamically determine new speakers to process
        NEW_SPEAKER_IDS_TO_PROCESS_AND_ADD = get_new_speakers_to_process(
            raw_data_repo_id=DATASET_REPO_ID, # Global raw dataset ID
            processed_data_repo_id=EXISTING_PROCESSED_DATASET_ID_ON_HUB
        )

        if not NEW_SPEAKER_IDS_TO_PROCESS_AND_ADD:
            print("No new speakers identified to process. If an existing dataset was provided, it might be re-pushed or the script might exit if nothing to do.")
            # The run_append_speaker_workflow will handle the case of no new speakers.
        else:
            print(f"Identified {len(NEW_SPEAKER_IDS_TO_PROCESS_AND_ADD)} new speakers for processing: {NEW_SPEAKER_IDS_TO_PROCESS_AND_ADD[:10]}...")

        run_append_speaker_workflow(
            existing_processed_dataset_id=EXISTING_PROCESSED_DATASET_ID_ON_HUB,
            new_speaker_ids_to_add=NEW_SPEAKER_IDS_TO_PROCESS_AND_ADD,
            combined_dataset_hub_id=COMBINED_DATASET_NEW_HUB_ID
        )

    elif args.mode == 'preprocess_config':
        print("\n--- Running in PREPROCESS_CONFIG mode ---")
        # This mode processes speakers from config.yaml and saves to a new dataset ID.
        # The output dataset ID for this mode comes from PROCESSED_HF_DATASET_REPO_FOR_UPLOAD global,
        # which can be dynamically suffixed based on speakers.
        # You might want args.output_dataset_id to override the base for PROCESSED_HF_DATASET_REPO_FOR_UPLOAD.
        
        _PROCESSED_HF_DATASET_REPO_FOR_UPLOAD_BASE = args.output_dataset_id # Use cmd line arg as base
        print(f"Base output dataset for preprocess_config mode: {_PROCESSED_HF_DATASET_REPO_FOR_UPLOAD_BASE}")

        target_speakers = get_speakers_to_filter_from_config(CONFIG_FILE_PATH_FOR_PREPROCESS)

        if target_speakers:
            speaker_suffix_list = [str(s).replace('/', '_').replace('\\', '_') for s in target_speakers]
            speaker_suffix = "_".join(sorted(speaker_suffix_list))
            if len(speaker_suffix) > 50: speaker_suffix = f"{len(target_speakers)}spkrs_h{hash(speaker_suffix)%1000}"
            PROCESSED_HF_DATASET_REPO_FOR_UPLOAD = f"{_PROCESSED_HF_DATASET_REPO_FOR_UPLOAD_BASE}-Filtered-{speaker_suffix}"
        else:
            PROCESSED_HF_DATASET_REPO_FOR_UPLOAD = f"{_PROCESSED_HF_DATASET_REPO_FOR_UPLOAD_BASE}-Filtered-All"
        
        print(f"Final output dataset name for this preprocessing run: {PROCESSED_HF_DATASET_REPO_FOR_UPLOAD}")

        run_preprocessing(speakers_to_process=target_speakers)

    else:
        print(f"Unknown mode: {args.mode}")