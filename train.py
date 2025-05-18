import os
import sys
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    AutoTokenizer,
    TrainerCallback
)
import soundfile as sf
import numpy as np
import yaml
import wandb
import torch
import random

# --- Configuration Loading ---
# Determine config file path (robust for different execution environments)
config_file_path = "/Orpheus/config.yaml" # Default for container
if not os.path.exists(config_file_path):
    # Try a path relative to the script file if __file__ is defined (standard execution)
    # Or relative to current working directory if __file__ is not defined (e.g. in some notebooks)
    script_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else "."
    config_file_path_rel = os.path.join(script_dir, "config.yaml")
    if os.path.exists(config_file_path_rel):
        config_file_path = config_file_path_rel
    else:
        # Fallback to current working directory if still not found
        config_file_path_cwd = os.path.join(os.getcwd(), "config.yaml")
        if os.path.exists(config_file_path_cwd):
            config_file_path = config_file_path_cwd
        else:
            print(f"FATAL: Config file not found. Tried:")
            print(f"  1. /Orpheus/config.yaml")
            print(f"  2. {config_file_path_rel}")
            print(f"  3. {config_file_path_cwd}")
            sys.exit(1)

print(f"Using configuration file: {config_file_path}")
try:
    with open(config_file_path, "r") as file:
        config = yaml.safe_load(file)
except Exception as e:
    print(f"Error reading config file '{config_file_path}': {e}")
    sys.exit(1)

# --- Read Configuration ---
dsn = config["TTS_dataset"]
model_name = config["model_name"]
run_name = config["run_name"]
project_name = config["project_name"]
base_repo_id = config["save_folder"] # Local output_dir for checkpoints
epochs = config["epochs"]
per_device_bs = config.get("per_device_train_batch_size", config.get("batch_size")) # Handle old 'batch_size' key
pad_token_id_from_config = config["pad_token"]
learning_rate = config["learning_rate"]
num_train_samples_config = config.get("num_train_samples", -1)

gradient_accumulation_steps = config.get("gradient_accumulation_steps", 1)
weight_decay = config.get("weight_decay", 0.0)
warmup_steps = config.get("warmup_steps", 0)
warmup_ratio = config.get("warmup_ratio", 0.0)
lr_scheduler_type = config.get("lr_scheduler_type", "linear") # Added from new config
gradient_checkpointing = config.get("gradient_checkpointing", False)

save_strategy = config.get("save_strategy", "steps")
save_steps = config.get("save_steps", 500)
save_total_limit = config.get("save_total_limit", None)
logging_steps = config.get("logging_steps", 100)
logging_first_step = config.get("logging_first_step", False)

# Mixed precision
bf16_enabled = config.get("bf16", False)
fp16_enabled = config.get("fp16", False) # Explicit fp16 setting

# Speaker Filtering
speakers_to_train_config = config.get("speakers_to_train", []) # Empty list if not specified

# Hugging Face Hub Push
push_enabled = config.get("push_to_hub", False)
hub_model_id = config.get("hub_model_id", None)
hub_private = config.get("hub_private_repo", True)

# Data Sanity Check
perform_data_check = config.get("perform_data_check", False)
num_samples_to_check = config.get("num_samples_to_check", 3)

# --- Constants and Setup ---
MODEL_MAX_LENGTH = 4096 # Orpheus default
# Special token IDs (from preprocess.py, ensure consistency)
tokeniser_length = 128256
start_of_speech_token = tokeniser_length + 1
end_of_speech_token = tokeniser_length + 2
start_of_human_token = tokeniser_length + 3
end_of_human_token = tokeniser_length + 4
start_of_ai_token = tokeniser_length + 5
end_of_ai_token = tokeniser_length + 6
AUDIO_CODE_BASE_OFFSET = tokeniser_length + 10



print("=" * 30)
print(f"--- Hyperparameters from Config ---")
print(f" >  Dataset: {dsn}")
print(f" >  Model Name: {model_name}")
print(f" >  Epochs: {epochs}")
print(f" >  Per-Device Batch Size: {per_device_bs}")
print(f" >  Gradient Accumulation Steps: {gradient_accumulation_steps}")
effective_bs_one_gpu = per_device_bs * gradient_accumulation_steps
print(f" >  Effective Batch Size (per GPU logic cycle): {effective_bs_one_gpu}")
print(f" >  Learning Rate: {learning_rate}, Scheduler: {lr_scheduler_type}")
print(f" >  Weight Decay: {weight_decay}")
print(f" >  Warmup Steps: {warmup_steps if warmup_steps > 0 else 'Disabled (or using ratio)'}")
print(f" >  Warmup Ratio: {warmup_ratio if warmup_ratio > 0 and warmup_steps == 0 else 'Disabled (or using steps)'}")
print(f" >  Gradient Checkpointing: {gradient_checkpointing}")
print(f" >  BF16: {bf16_enabled}, FP16: {fp16_enabled}")
print(f" >  Save Strategy: {save_strategy}, Steps: {save_steps}, Total Limit: {save_total_limit}")
print(f" >  Logging Steps: {logging_steps}, Log First Step: {logging_first_step}")
print(f" >  Pad Token ID (from config): {pad_token_id_from_config}")
print(f" >  MODEL_MAX_LENGTH: {MODEL_MAX_LENGTH}")
print(f" >  Target speakers for training: {speakers_to_train_config if speakers_to_train_config else 'ALL'}")
print(f" >  Number of train samples (from config): {'ALL' if num_train_samples_config == -1 else num_train_samples_config}")
print("=" * 30)

# --- Determine Mixed Precision ---
use_bf16 = False
use_fp16 = False
if bf16_enabled and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
    use_bf16 = True
    print("BF16 is enabled and supported.")
elif fp16_enabled and torch.cuda.is_available():
    use_fp16 = True
    print("FP16 is enabled (BF16 not enabled or not supported).")
else:
    print("Mixed precision (BF16/FP16) is not enabled or not supported by hardware/config.")


print("--- Loading Base Model and Tokenizer ---")
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    model_load_kwargs = {}
    if use_bf16:
        model_load_kwargs["torch_dtype"] = torch.bfloat16
    elif use_fp16: # only if bf16 not used
        model_load_kwargs["torch_dtype"] = torch.float16

    # Flash Attention 2 (requires PyTorch 2.0+, CUDA, and Ampere+ GPU)
    # Check for PyTorch version supporting attn_implementation
    pt_version = torch.__version__
    if tuple(map(int, pt_version.split('.')[:2])) >= (2, 0) and torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
        model_load_kwargs["attn_implementation"] = "flash_attention_2"
        print("Attempting to use Flash Attention 2.")
    else:
        print("Flash Attention 2 not available or not optimal for this GPU/PyTorch version. Using default attention.")

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_load_kwargs)

except Exception as e:
    print(f"Error loading base model/tokenizer '{model_name}': {e}")
    sys.exit(1)

# --- Pad Token Handling ---
if tokenizer.pad_token_id is None:
    print(f"Tokenizer pad_token_id is None. Setting to {pad_token_id_from_config} from config.")
    tokenizer.add_special_tokens({'pad_token': str(pad_token_id_from_config)}) # Add as a string
    tokenizer.pad_token_id = pad_token_id_from_config # Assign the integer ID
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id # Ensure model config is also aligned
    print(f"Model token embeddings resized. New vocab size: {len(tokenizer)}")
elif tokenizer.pad_token_id != pad_token_id_from_config:
    print(f"Warning: Tokenizer pad_token_id ({tokenizer.pad_token_id}) differs from config ({pad_token_id_from_config}).")
    print(f"Using pad_token_id from config: {pad_token_id_from_config}.")
    tokenizer.pad_token_id = pad_token_id_from_config
    model.config.pad_token_id = tokenizer.pad_token_id
else:
    print(f"Tokenizer pad_token_id ({tokenizer.pad_token_id}) matches config ({pad_token_id_from_config}).")
    model.config.pad_token_id = tokenizer.pad_token_id


# --- Dataset Loading ---
print(f"\n--- Loading Processed Dataset: {dsn} ---")
try:
    # The dataset from preprocess.py should have 'input_ids', 'labels', 'attention_mask'
    # and 'speaker_id' (if kept for filtering).
    ds_full = load_dataset(dsn, split="train", trust_remote_code=True) # trust_remote_code for datasets with custom loading scripts
    print(f"Full dataset '{dsn}' loaded with {len(ds_full)} samples.")
    print(f"Columns in loaded dataset: {ds_full.column_names}")
    if not all(col in ds_full.column_names for col in ['input_ids', 'labels', 'attention_mask']):
        print(f"FATAL: Dataset {dsn} is missing required columns: 'input_ids', 'labels', 'attention_mask'.")
        sys.exit(1)
except Exception as e:
    print(f"Error loading dataset '{dsn}': {e}")
    sys.exit(1)

# --- Speaker Filtering ---
ds_after_speaker_filter = ds_full
if speakers_to_train_config and isinstance(speakers_to_train_config, list) and len(speakers_to_train_config) > 0:
    print(f"\n--- Filtering dataset for specific speakers: {speakers_to_train_config} ---")
    if "speaker_id" not in ds_full.column_names:
        print(f"ERROR: 'speaker_id' column not found in dataset {dsn}, but 'speakers_to_train' is specified in config.")
        print("Cannot filter by speaker. Please ensure 'speaker_id' is present in the processed dataset.")
        sys.exit(1)
    
    initial_speaker_filter_len = len(ds_full)
    # Using a set for faster lookups if speakers_to_train_config is large
    target_speaker_set = set(speakers_to_train_config)
    ds_after_speaker_filter = ds_full.filter(
        lambda example: example["speaker_id"] in target_speaker_set,
        num_proc=os.cpu_count() or 1 # Use multiple processes for filtering if available
    )
    filtered_speaker_count = initial_speaker_filter_len - len(ds_after_speaker_filter)
    print(f"Filtered out {filtered_speaker_count} samples from other speakers.")
    print(f"Dataset size after speaker filtering: {len(ds_after_speaker_filter)}")
    if len(ds_after_speaker_filter) == 0:
        print("ERROR: Dataset is empty after filtering for specified speakers. Check speaker_ids in config and dataset.")
        sys.exit(1)
else:
    print("\n--- No specific speakers to filter for, using all speakers in the dataset. ---")


# --- Length Filtering ---
print(f"\n--- Filtering dataset by MODEL_MAX_LENGTH ({MODEL_MAX_LENGTH}) ---")
initial_len_filter_len = len(ds_after_speaker_filter)
ds_after_len_filter = ds_after_speaker_filter.filter(
    lambda x: len(x["input_ids"]) <= MODEL_MAX_LENGTH,
    num_proc=os.cpu_count() or 1
)
filtered_len_count = initial_len_filter_len - len(ds_after_len_filter)
if filtered_len_count > 0:
    print(f"Filtered out {filtered_len_count} samples because 'input_ids' length exceeded MODEL_MAX_LENGTH.")
print(f"Dataset size after length filtering: {len(ds_after_len_filter)}")
if len(ds_after_len_filter) == 0:
    print("ERROR: Dataset is empty after length filtering."); sys.exit(1)


# --- Subsetting (num_train_samples from config) ---
ds_to_train = ds_after_len_filter
if isinstance(num_train_samples_config, int) and num_train_samples_config > 0 and num_train_samples_config < len(ds_after_len_filter):
    print(f"\n--- Selecting {num_train_samples_config} random samples for training (from config) ---")
    # Shuffle before selecting to get a random subset
    ds_shuffled = ds_after_len_filter.shuffle(seed=42) # Fixed seed for reproducibility
    ds_to_train = ds_shuffled.select(range(num_train_samples_config))
    print(f"Using a subset of {len(ds_to_train)} samples for this training run.")
else:
    print(f"\n--- Using all {len(ds_after_len_filter)} available samples (after speaker/length filtering) for training. ---")

print(f"Final dataset size for this run: {len(ds_to_train)}")
if len(ds_to_train) == 0:
    print("ERROR: No samples selected for training after all filtering steps."); sys.exit(1)


# --- Data Sanity Check ---
if perform_data_check and len(ds_to_train) > 0:
    print(f"\n--- Performing Data Sanity Check on {min(num_samples_to_check, len(ds_to_train))} samples ---")
    actual_samples_to_inspect_count = min(num_samples_to_check, len(ds_to_train))
    indices_to_check = random.sample(range(len(ds_to_train)), actual_samples_to_inspect_count)

    for i, sample_idx_in_current_ds in enumerate(indices_to_check):
        try:
            item = ds_to_train[sample_idx_in_current_ds]
            print(f"\n--- Sample {i+1} (from current training_ds, index: {sample_idx_in_current_ds}) ---")
            input_ids = item.get("input_ids", [])
            labels = item.get("labels", []) # Should be same as input_ids for Causal LM at this stage
            attn_mask = item.get("attention_mask", [])
            
            speaker_id_from_sample = item.get("speaker_id", "N/A_speaker_id_col_missing") 
            # sentence_id_from_sample = item.get("sentence_id", "N/A_sentence_id_col_missing") # If you need to check this

            print(f"  Speaker ID (from data field): {speaker_id_from_sample}")
            print(f"  Length input_ids: {len(input_ids)}")
            # print(f"  Length labels:    {len(labels)}") # Redundant if checking equality
            # print(f"  Length attn_mask: {len(attn_mask)}")

            if not (len(input_ids) == len(labels) == len(attn_mask)):
                 print("  >>> ALERT: Length mismatch between input_ids/labels/attn_mask!")
            if not input_ids:
                 print("  >>> ALERT: input_ids is empty for this sample!")
                 continue # Skip further checks for this empty sample
            if len(input_ids) > MODEL_MAX_LENGTH: 
                 print(f"  >>> ALERT: Length ({len(input_ids)}) exceeds MODEL_MAX_LENGTH ({MODEL_MAX_LENGTH})! (Should have been filtered)")
            if not all(x == 1 for x in attn_mask): # Data from preprocess.py should have all 1s before collator
                 print(f"  >>> INFO: 'attention_mask' contains zeros before collation. Sum: {sum(attn_mask)}/{len(attn_mask)}")
            if input_ids != labels:
                 print(f"  >>> ALERT: 'input_ids' and 'labels' are not identical for this sample before collation.")


            # print(f"  Input_ids (first 20): {input_ids[:20]}")
            # decoded_full_preview = tokenizer.decode(input_ids[:50], skip_special_tokens=False) 
            # print(f"  Decoded Full input_ids (first 50 tokens approx): '{decoded_full_preview}...'")

            try:
                # Find first occurrences, assuming only one SOH/EOH block per sample
                start_idx_text_human = input_ids.index(start_of_human_token) + 1
                end_idx_text_human = input_ids.index(end_of_human_token)
                
                text_tokens_segment = input_ids[start_idx_text_human : end_idx_text_human]
                decoded_text_segment = tokenizer.decode(text_tokens_segment, skip_special_tokens=True) # skip_special_tokens for readability
                print(f"  Decoded Text Segment (SOH to EOH): '{decoded_text_segment}'")

                if speaker_id_from_sample != "N/A_speaker_id_col_missing":
                    expected_prefix = f"{speaker_id_from_sample}:"
                    if not decoded_text_segment.startswith(expected_prefix):
                        print(f"    >>> VERIFICATION WARNING: Decoded text segment does not start with expected speaker prefix '{expected_prefix}'. Found: '{decoded_text_segment[:len(expected_prefix)+10]}...'")
                else:
                    # This case should not happen if speaker filtering worked or if 'speaker_id' is required by the pipeline
                    print("    INFO: 'speaker_id' column not found in this dataset sample. Cannot verify speaker prefix in text.")

            except ValueError: # .index() fails
                print("  >>> ALERT: Could not find SOH or EOH tokens in input_ids. Structure might be incorrect.")
            except Exception as decode_e:
                 print(f"  Error during sanity check decoding: {decode_e}")
        except Exception as sample_e:
            print(f"\nError processing sample (index {sample_idx_in_current_ds}) for sanity check: {sample_e}")
    print("--- End Data Sanity Check ---\n")


# --- Manual Padding Data Collator ---
class ManualPaddingDataCollator:
    def __init__(self, tokenizer, padding_value_map=None, model_max_length=None):
        self.tokenizer = tokenizer
        self.padding_value_map = padding_value_map if padding_value_map is not None else {
            "input_ids": self.tokenizer.pad_token_id,
            "labels": -100, # Standard for ignoring loss on padded labels
            "attention_mask": 0, # Padded positions in attention_mask are 0
        }
        self.model_max_length = model_max_length
        if self.tokenizer.pad_token_id is None:
            raise ValueError("DataCollator: Tokenizer needs a pad_token_id.")
        if self.model_max_length is None:
            print("Warning: DataCollator: model_max_length not set. Will pad to max length in batch.")
        
        print(f"DataCollator initialized: pad_token_id={self.tokenizer.pad_token_id}, label_pad_value={self.padding_value_map['labels']}, model_max_length={self.model_max_length}")

    def __call__(self, features):
        valid_features = [f for f in features if isinstance(f, dict) and f.get("input_ids")]
        if not valid_features:
            print("CRITICAL WARNING (Collator): Batch contains no valid features. Returning empty dict.")
            return {}
        
        features = valid_features
        batch_max_seq_len_in_batch = max(len(f["input_ids"]) for f in features)
        
        # Pad to model_max_length if set and batch_max is less, else pad to batch_max (up to model_max_length)
        if self.model_max_length:
            pad_to_length = min(batch_max_seq_len_in_batch, self.model_max_length)
            # If all sequences in batch are shorter than model_max_length, we still pad them up to model_max_length
            # to ensure consistent tensor shapes across batches for some strategies (like gradient_checkpointing)
            # However, standard behavior is to pad to max in batch UP TO model_max_length.
            # Let's stick to: pad to max_in_batch, but truncate at model_max_length.
            # The truncation to model_max_length happens to sequences *before* this collator due to dataset.filter.
            # So, batch_max_seq_len_in_batch will always be <= self.model_max_length.
            # Thus, we pad to batch_max_seq_len_in_batch.
            # If we want all batches to have exactly model_max_length (even if items are shorter), uncomment next line:
            # pad_to_length = self.model_max_length
            pad_to_length = batch_max_seq_len_in_batch # Max length in this current batch
        else:
            pad_to_length = batch_max_seq_len_in_batch

        batch = {}
        keys_to_pad = ["input_ids", "labels", "attention_mask"]

        for key in keys_to_pad:
            if key not in self.padding_value_map: continue
            padding_value = self.padding_value_map[key]
            sequences = [f.get(key, []) for f in features]
            padded_sequences = []
            for i, seq in enumerate(sequences):
                if not isinstance(seq, list):
                    raise TypeError(f"Collator: Invalid type for key '{key}' in feature {i}. Expected list, got {type(seq)}.")
                
                # Sequences should already be truncated by the earlier filter.
                # Here, we just pad them to `pad_to_length`.
                seq_current = seq[:pad_to_length] # Ensure it's not longer just in case
                needed_padding = pad_to_length - len(seq_current)
                padded_seq = seq_current + ([padding_value] * needed_padding)
                padded_sequences.append(padded_seq)
            
            try:
                if not all(len(ps) == pad_to_length for ps in padded_sequences):
                    raise ValueError(f"Collator: Inconsistent lengths for key '{key}'. Target: {pad_to_length}, Got: {[len(ps) for ps in padded_sequences]}")
                batch[key] = torch.tensor(padded_sequences, dtype=torch.long)
            except Exception as e:
                print(f"ERROR creating tensor for key '{key}' in collator: {e}")
                raise
        return batch

data_collator = ManualPaddingDataCollator(tokenizer=tokenizer, model_max_length=MODEL_MAX_LENGTH)


# --- WandB Setup ---
print("\n--- Initializing WandB ---")
# Check if WANDB_API_KEY is set, or if WANDB_MODE is disabled
wandb_mode_to_use = "online"
if not os.environ.get("WANDB_API_KEY"):
    print("WANDB_API_KEY not found in environment. Setting WandB mode to 'disabled'.")
    wandb_mode_to_use = "disabled"
if os.environ.get("WANDB_MODE") == "disabled": # Explicit disable overrides API key presence
    print("WANDB_MODE environment variable is set to 'disabled'.")
    wandb_mode_to_use = "disabled"

try:
    wandb.init(project=project_name, name=run_name, config=config, mode=wandb_mode_to_use)
    if wandb.run.mode != "disabled":
        print(f"WandB initialized successfully for project='{project_name}', run='{run_name}'.")
    else:
        print("WandB is running in disabled mode.")
except Exception as e:
    print(f"Error initializing WandB: {e}. Training will continue without WandB logging.")
    wandb.init(mode="disabled") # Ensure wandb.run exists even if disabled

# --- Training Arguments ---
print("\n--- Setting Training Arguments ---")
output_dir_local = f"./{base_repo_id}" # Local directory for checkpoints
training_args = TrainingArguments(
    output_dir=output_dir_local,
    dataloader_num_workers=2, # Adjust based on CPU cores and I/O bandwidth
    overwrite_output_dir=True,
    do_train=True,
    # do_eval=False, # No eval_dataset specified

    num_train_epochs=epochs,
    per_device_train_batch_size=per_device_bs,
    gradient_accumulation_steps=gradient_accumulation_steps,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    warmup_steps=warmup_steps if warmup_steps > 0 else 0,
    warmup_ratio=warmup_ratio if warmup_ratio > 0 and warmup_steps == 0 else 0.0,
    lr_scheduler_type=lr_scheduler_type,

    logging_dir=f"{output_dir_local}/logs",
    logging_strategy="steps",
    logging_steps=logging_steps,
    logging_first_step=logging_first_step,

    save_strategy=save_strategy,
    save_steps=save_steps,
    save_total_limit=save_total_limit,
    # load_best_model_at_end=False, # Requires eval_dataset and metric_for_best_model

    bf16=use_bf16,
    fp16=use_fp16, # Use determined fp16 status
    
    gradient_checkpointing=gradient_checkpointing,
    # ddp_find_unused_parameters=False, # Only if using DDP and facing related issues
    
    report_to="wandb" if wandb.run and wandb.run.mode != "disabled" else "none",
    remove_unused_columns=True, # Trainer will remove columns not used by model's forward (e.g., 'speaker_id' if still present)
    max_steps=-1, # No limit by steps, epochs control overall duration unless overridden
)

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs): # Add **kwargs
        outputs = model(**inputs) # Unpacks dict into keyword arguments for model.forward
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss


class GenerateSamplesCallback(TrainerCallback):
    def __init__(self, tokenizer, test_prompts, speaker_ids, output_dir="eval_samples", wandb_run=None, log_to_wandb=True, generate_every_n_steps=None):
        super().__init__()
        self.tokenizer = tokenizer
        self.test_prompts = test_prompts # List of text prompts
        # Support single speaker or list of speakers
        self.speaker_ids = [speaker_ids] if isinstance(speaker_ids, str) else speaker_ids
        print(f"GenerateSamplesCallback initialized with {len(self.speaker_ids)} speakers: {self.speaker_ids}")
        
        self.output_dir_base = output_dir
        self.wandb_run = wandb_run
        self.log_to_wandb = log_to_wandb and (wandb_run is not None and wandb_run.mode != "disabled")
        self.generate_every_n_steps = generate_every_n_steps # New: generate every N steps

        # Ensure special tokens are available
        self.SOH = tokeniser_length + 3
        self.EOH = tokeniser_length + 4
        self.SOA = tokeniser_length + 5
        self.SOS = tokeniser_length + 1 # Start of Speech for generation target

        if not os.path.exists(self.output_dir_base):
            os.makedirs(self.output_dir_base, exist_ok=True)

    def _generate_audio(self, model, prompt_text, speaker_id, step_or_epoch):
        model.eval() # Set model to evaluation mode
        device = model.device

        # Construct the input prompt for the model, including the speaker tag
        # This MUST match the format used in preprocess.py for training data
        full_prompt_text = f"{speaker_id}: {prompt_text}"
        
        input_ids_list = (
            [self.SOH] +
            self.tokenizer.encode(full_prompt_text, add_special_tokens=False) +
            [self.EOH, self.SOA, self.SOS] # Model should generate from <SOS>
        )
        input_ids = torch.tensor([input_ids_list], device=device)

        print(f"\nGenerating sample for: '{prompt_text}' (Speaker: {speaker_id}) at step/epoch {step_or_epoch}")
        
        with torch.no_grad():
            # Adjust generation parameters as needed for Orpheus
            # max_new_tokens might need to be quite large to capture enough audio frames
            # For Orpheus, we expect it to generate SNAC codes and then EOSpeech, EOA
            generated_ids = model.generate(
                input_ids,
                max_new_tokens=1024, # Max SNAC tokens to generate (7 tokens per ~32ms frame)
                                     # 1024 SNAC tokens = ~146 frames = ~4.6 seconds. Adjust as needed.
                num_beams=1,         # Greedy decoding is usually fine for TTS audio codes
                do_sample=False,     # Deterministic generation for audio codes
                pad_token_id=self.tokenizer.pad_token_id, # Important for generation
                eos_token_id=end_of_speech_token # Stop when <EOSpeech> is generated
                # You might also need to add end_of_ai_token if the model is trained to produce it strictly after EOSpeech
            )
        
        # Extract only the generated part (after the prompt and <SOS>)
        generated_sequence = generated_ids[0][input_ids.shape[1]:]
        
        # Find the actual SNAC codes by looking for tokens in the SNAC range
        # and stopping at end_of_speech_token or end_of_ai_token
        snac_codes = []
        for token_id in generated_sequence.tolist():
            if token_id == end_of_speech_token or token_id == end_of_ai_token:
                break
            # Check if token is within any of the 7 SNAC codebook ranges
            # (AUDIO_CODE_BASE_OFFSET to AUDIO_CODE_BASE_OFFSET + 4095*7 -1 approx)
            # A simpler check: if it's above the text tokenizer vocab and special tokens,
            # and below a very high number, it's likely an audio code.
            # This check might need refinement based on your exact token ID ranges.
            if token_id >= AUDIO_CODE_BASE_OFFSET and token_id < (AUDIO_CODE_BASE_OFFSET + 4096 * 7): # 7 codebooks for SNAC
                snac_codes.append(token_id)
        
        if not snac_codes:
            print(f"Warning: No SNAC codes generated or extracted for speaker {speaker_id}.")
            return None, None

        # Decode SNAC codes to audio waveform
        # This requires a SNAC decoder (or your trained SNAC model's decoder part)
        # For now, let's assume you have a function `decode_snac_to_waveform`
        # If not, you'll need to implement or find one.
        # A placeholder: we can't directly convert without SNAC decoder.
        # We'll just save the SNAC codes for now.
        
        # If you had a SNAC decoder:
        # snac_decoder = SNAC.from_pretrained(SNAC_MODEL_NAME).to(device).eval() # Or reuse if available
        # # Format snac_codes for decoder: list of 3 tensors (codes_lvl0, codes_lvl1, codes_lvl2)
        # # This part is complex as you need to de-interleave the 7 codes per frame back into 3 levels.
        # # For simplicity, this example won't implement the full de-interleaving and SNAC decoding.
        # # waveform_tensor = snac_decoder.decode(formatted_snac_codes_for_decoder)
        # # audio_data = waveform_tensor.squeeze().cpu().numpy()
        # # sf.write(filepath, audio_data, TARGET_AUDIO_SAMPLING_RATE)

        # For now, just log that we generated codes
        print(f"Generated {len(snac_codes)} SNAC codes for speaker {speaker_id}.")
        
        # Create a dummy audio file or log to WandB
        # This is a placeholder as true audio generation from SNAC codes requires the SNAC decoder part.
        # We can log the text and the fact that codes were generated.
        filename_safe_prompt = "".join(c if c.isalnum() else "_" for c in prompt_text[:30])
        # Create a dummy .txt file with snac codes for now
        snac_codes_filepath = os.path.join(self.output_dir_base, f"step_{step_or_epoch}_spk_{speaker_id}_{filename_safe_prompt}_snac_codes.txt")
        with open(snac_codes_filepath, "w") as f:
            f.write(" ".join(map(str, snac_codes)))
        print(f"Saved SNAC codes to {snac_codes_filepath}")

        if self.log_to_wandb:
            # Since we don't have easy audio decoding here, log the text and a placeholder.
            # If you integrate SNAC decoding, you'd use wandb.Audio
            self.wandb_run.log({
                f"eval_sample_step_{step_or_epoch}/spk_{speaker_id}_prompt_{filename_safe_prompt}_snac_codes_length": len(snac_codes),
                # "eval_sample/audio": wandb.Audio(audio_data, caption=prompt_text, sample_rate=TARGET_AUDIO_SAMPLING_RATE) # If you had audio_data
            }, step=step_or_epoch) # Use trainer's global step

        return snac_codes_filepath, len(snac_codes)


    def on_save(self, args: TrainingArguments, state, control, model=None, tokenizer=None, **kwargs):
        # This method is called when a checkpoint is saved.
        # 'model' and 'tokenizer' are passed by the Trainer.
        if state.is_world_process_zero: # Ensure generation happens only on main process
            print(f"\n--- Callback: Generating samples after saving checkpoint at step {state.global_step} ---")
            for speaker_id in self.speaker_ids:
                print(f"Generating samples for speaker {speaker_id}")
                for prompt in self.test_prompts:
                    self._generate_audio(model, prompt, speaker_id, state.global_step)
            model.train() # Set model back to training mode

    def on_step_end(self, args: TrainingArguments, state, control, model=None, tokenizer=None, **kwargs):
        if self.generate_every_n_steps and state.is_world_process_zero and (state.global_step > 0 and state.global_step % self.generate_every_n_steps == 0) :
            if state.global_step % args.save_steps == 0 : # Don't generate if already generating on save
                return

            print(f"\n--- Callback: Generating samples at step {state.global_step} (every {self.generate_every_n_steps} steps) ---")
            current_model_training_state = model.training # Save current model state
            model.eval()
            for speaker_id in self.speaker_ids:
                print(f"Generating samples for speaker {speaker_id}")
                for prompt in self.test_prompts:
                    self._generate_audio(model, prompt, speaker_id, state.global_step)
            if current_model_training_state: # Restore if it was training
                model.train()



callbacks_to_use = []
if config.get("generate_test_samples_on_save", False) or config.get("generate_test_samples_every_n_steps", 0) > 0:
    test_prompts_from_config = config.get("test_generation_prompts", [])
    
    # Determine which speakers to generate samples for
    target_speakers_for_gen = []
    generate_for_all = config.get("generate_for_all_speakers", False)
    
    # If generate_for_all_speakers is true, use all speakers being trained
    if generate_for_all and speakers_to_train_config:
        target_speakers_for_gen = speakers_to_train_config
        print(f"Will generate samples for all speakers being trained: {target_speakers_for_gen}")
    # Otherwise, check if a single target speaker is specified in config
    elif config.get("target_speaker_for_generation"):
        target_speakers_for_gen = [config.get("target_speaker_for_generation")]
        print(f"Will generate samples for single speaker specified in config: {target_speakers_for_gen}")
    # If one speaker is being trained, use that 
    elif speakers_to_train_config and len(speakers_to_train_config) == 1:
        target_speakers_for_gen = speakers_to_train_config
        print(f"Will generate samples for the one speaker being trained: {target_speakers_for_gen}")
    
    if test_prompts_from_config and target_speakers_for_gen:
        print(f"Initializing GenerateSamplesCallback for {len(target_speakers_for_gen)} speaker(s)")
        generate_callback = GenerateSamplesCallback(
            tokenizer=tokenizer, # The main tokenizer
            test_prompts=test_prompts_from_config,
            speaker_ids=target_speakers_for_gen,
            output_dir=os.path.join(output_dir_local, "generated_eval_samples"), # Save samples inside checkpoint dir
            wandb_run=wandb.run if wandb.run and wandb.run.mode != "disabled" else None, # Pass wandb run object
            log_to_wandb=True, # Or control this via config
            generate_every_n_steps=config.get("generate_test_samples_every_n_steps", 0)
        )
        callbacks_to_use.append(generate_callback)
    elif not test_prompts_from_config:
        print("Warning: Test sample generation enabled in config, but 'test_generation_prompts' is empty.")
    elif not target_speakers_for_gen:
        print("Warning: Test sample generation enabled, but no target speakers found. Please set 'generate_for_all_speakers: true' or 'target_speaker_for_generation' in config.")



# --- Initialize Trainer ---
print("\n--- Initializing Trainer ---")
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=ds_to_train,
    data_collator=data_collator,
    tokenizer=tokenizer, # Pass tokenizer here for the Trainer itself
    callbacks=callbacks_to_use # Add the custom callback here
)

# --- Start Training ---
num_gpus_detected = torch.cuda.device_count() if torch.cuda.is_available() else 0
print(f"\n--- Starting Training ({num_gpus_detected} GPU(s) detected by PyTorch) ---")
if num_gpus_detected > 0:
    print(f"Effective total batch size across all GPUs: {effective_bs_one_gpu * num_gpus_detected}")
else:
    print(f"Effective total batch size (CPU): {effective_bs_one_gpu}")

print(f"Model will be trained for {epochs} epochs on {len(ds_to_train)} samples.")
print(f"Checkpoints will be saved in: {output_dir_local}")
if push_enabled and hub_model_id:
    print(f"Final model will be pushed to Hugging Face Hub at: {hub_model_id}")

training_successful = False
try:
    train_result = trainer.train() # Can add resume_from_checkpoint=True or path if needed
    print("\n--- Training finished successfully! ---")
    trainer.save_model() # Save final model state explicitly
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state() # Save optimizer, scheduler, etc.
    training_successful = True
except Exception as e:
    print(f"\n--- Training failed! ---")
    import traceback
    print(traceback.format_exc())
    if os.path.exists(output_dir_local) and any(os.scandir(output_dir_local)):
        print(f"Attempting to save model/state from partially completed training to {output_dir_local}...")
        try:
            trainer.save_model()
            trainer.save_state()
            print("Partial model/state saved.")
        except Exception as save_e:
            print(f"Could not save partial model/state: {save_e}")

# --- Push to Hub ---
if push_enabled and hub_model_id:
    if training_successful:
        print("\n--- Attempting to Push Model and Tokenizer to Hub ---")
        try:
            commit_msg = f"Training complete for run: {run_name} ({epochs} epochs, LR {learning_rate})"
            if hasattr(train_result, 'metrics') and 'train_loss' in train_result.metrics: # metrics might not exist if training fails early
                commit_msg += f". Final train_loss: {train_result.metrics['train_loss']:.4f}"
            
            # trainer.push_to_hub() pushes content of training_args.output_dir
            trainer.push_to_hub(
                repo_id=hub_model_id,
                private=hub_private,
                commit_message=commit_msg
            )
            print(f"--- Model pushed successfully to https://huggingface.co/{hub_model_id} ---")
        except Exception as e:
            print(f"\n--- Error pushing model: {e} ---")
            print("  Common issues: `huggingface-cli login` needed, `git-lfs` not installed, or no write access to repo.")
            print(f"  Model checkpoints are still available locally at: {output_dir_local}")
    else:
        print("\n--- Skipping Hub push (training was not successful or was interrupted). ---")
        print(f"  Model checkpoints (if any saved) are at: {output_dir_local}")
else:
    print("\n--- Skipping Hub push (push_to_hub: false or hub_model_id not set in config). ---")
    if training_successful:
        print(f"Final model checkpoint saved locally at: {output_dir_local}")

if wandb.run and wandb.run.mode != "disabled":
    wandb.finish()
print("\n--- Script finished ---")
