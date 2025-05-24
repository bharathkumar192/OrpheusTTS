import os
import torch
import torchaudio
import torchaudio.transforms as T
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from snac import SNAC 
import soundfile as sf
import numpy as np
import argparse
import time
import re 

# --- Configuration ---
# Match these with your preprocess.py and training setup
SNAC_MODEL_NAME = "hubertsiuzdak/snac_24khz"
TARGET_AUDIO_SAMPLING_RATE = 24000

# Special Token IDs (ensure consistency with preprocess.py/train.py)
tokeniser_length = 128256
start_of_speech_token = tokeniser_length + 1
end_of_speech_token = tokeniser_length + 2
start_of_human_token = tokeniser_length + 3
end_of_human_token = tokeniser_length + 4
start_of_ai_token = tokeniser_length + 5
end_of_ai_token = tokeniser_length + 6
AUDIO_CODE_BASE_OFFSET = tokeniser_length + 10
PAD_TOKEN_ID = 128263 # Make sure this matches your config.yaml and tokenizer setting

# SNAC codebook offsets (derived from AUDIO_CODE_BASE_OFFSET and structure in preprocess.py)
# Frame structure: [L0, L1a, L2a, L2b, L1b, L2c, L2d]
SNAC_OFFSETS = [
    AUDIO_CODE_BASE_OFFSET + 0 * 4096, # Offset for L0 code (index 0)
    AUDIO_CODE_BASE_OFFSET + 1 * 4096, # Offset for L1a code (index 1)
    AUDIO_CODE_BASE_OFFSET + 2 * 4096, # Offset for L2a code (index 2)
    AUDIO_CODE_BASE_OFFSET + 3 * 4096, # Offset for L2b code (index 3)
    AUDIO_CODE_BASE_OFFSET + 4 * 4096, # Offset for L1b code (index 4)
    AUDIO_CODE_BASE_OFFSET + 5 * 4096, # Offset for L2c code (index 5)
    AUDIO_CODE_BASE_OFFSET + 6 * 4096, # Offset for L2d code (index 6)
]

def deinterleave_snac_codes(snac_codes_flat, device):
    """
    De-interleaves the flat list of 7 SNAC codes per frame back into
    the 3 separate code levels expected by the SNAC decoder.
    """
    if not snac_codes_flat or len(snac_codes_flat) % 7 != 0:
        print(f"Warning: Invalid flat SNAC code list length ({len(snac_codes_flat)}). Must be multiple of 7 or non-empty.")
        return None, None, None

    num_frames = len(snac_codes_flat) // 7
    codes_lvl0 = []
    codes_lvl1 = []
    codes_lvl2 = []

    for i in range(0, len(snac_codes_flat), 7):
        try:
            # Ensure codes are within their expected codebook range (0-4095 after offset subtraction)
            if not (SNAC_OFFSETS[0] <= snac_codes_flat[i] < SNAC_OFFSETS[0] + 4096): raise ValueError(f"L0 code {snac_codes_flat[i]} out of range for expected offset {SNAC_OFFSETS[0]}")
            if not (SNAC_OFFSETS[1] <= snac_codes_flat[i+1] < SNAC_OFFSETS[1] + 4096): raise ValueError(f"L1a code {snac_codes_flat[i+1]} out of range for expected offset {SNAC_OFFSETS[1]}")
            if not (SNAC_OFFSETS[2] <= snac_codes_flat[i+2] < SNAC_OFFSETS[2] + 4096): raise ValueError(f"L2a code {snac_codes_flat[i+2]} out of range for expected offset {SNAC_OFFSETS[2]}")
            if not (SNAC_OFFSETS[3] <= snac_codes_flat[i+3] < SNAC_OFFSETS[3] + 4096): raise ValueError(f"L2b code {snac_codes_flat[i+3]} out of range for expected offset {SNAC_OFFSETS[3]}")
            if not (SNAC_OFFSETS[4] <= snac_codes_flat[i+4] < SNAC_OFFSETS[4] + 4096): raise ValueError(f"L1b code {snac_codes_flat[i+4]} out of range for expected offset {SNAC_OFFSETS[4]}")
            if not (SNAC_OFFSETS[5] <= snac_codes_flat[i+5] < SNAC_OFFSETS[5] + 4096): raise ValueError(f"L2c code {snac_codes_flat[i+5]} out of range for expected offset {SNAC_OFFSETS[5]}")
            if not (SNAC_OFFSETS[6] <= snac_codes_flat[i+6] < SNAC_OFFSETS[6] + 4096): raise ValueError(f"L2d code {snac_codes_flat[i+6]} out of range for expected offset {SNAC_OFFSETS[6]}")
            
            codes_lvl0.append(snac_codes_flat[i]   - SNAC_OFFSETS[0])
            codes_lvl1.append(snac_codes_flat[i+1] - SNAC_OFFSETS[1])
            codes_lvl2.append(snac_codes_flat[i+2] - SNAC_OFFSETS[2])
            codes_lvl2.append(snac_codes_flat[i+3] - SNAC_OFFSETS[3])
            codes_lvl1.append(snac_codes_flat[i+4] - SNAC_OFFSETS[4])
            codes_lvl2.append(snac_codes_flat[i+5] - SNAC_OFFSETS[5])
            codes_lvl2.append(snac_codes_flat[i+6] - SNAC_OFFSETS[6])
        except (IndexError, ValueError) as e:
            print(f"Error processing SNAC codes at block starting index {i}: {e}. Invalid sequence for decoding.")
            return None, None, None

    try:
        codes_lvl0_tensor = torch.tensor(codes_lvl0, dtype=torch.long, device=device).unsqueeze(0)
        codes_lvl1_tensor = torch.tensor(codes_lvl1, dtype=torch.long, device=device).unsqueeze(0)
        codes_lvl2_tensor = torch.tensor(codes_lvl2, dtype=torch.long, device=device).unsqueeze(0)

        if codes_lvl0_tensor.shape[1] != num_frames: raise ValueError("L0 shape mismatch")
        if codes_lvl1_tensor.shape[1] != 2 * num_frames: raise ValueError("L1 shape mismatch")
        if codes_lvl2_tensor.shape[1] != 4 * num_frames: raise ValueError("L2 shape mismatch")

    except Exception as e:
        print(f"Error creating tensors during de-interleaving: {e}")
        return None, None, None

    return codes_lvl0_tensor, codes_lvl1_tensor, codes_lvl2_tensor

def run_inference_for_single_checkpoint(repo_id, subfolder, prompts, target_speaker_ids, output_dir, device="cuda", inference_batch_size=50, use_auth_token=None):
    if subfolder:
        checkpoint_name_safe = subfolder.replace("/", "_").replace("\\", "_")
        model_display_name = f"{repo_id}/{subfolder}"
    else:
        checkpoint_name_safe = os.path.basename(repo_id).replace("/", "_").replace("\\", "_") + "_base" # Mark base explicitly
        model_display_name = repo_id
    
    print(f"\n--- Starting Inference for Model: {model_display_name} ---")
    print(f"  Repo ID: {repo_id}, Subfolder: {subfolder if subfolder else 'N/A (root)'}")
    print(f"  Outputting to: {output_dir}")
    print(f"  Using device: {device}")


    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"  Created output directory: {output_dir}")

    print(f"  Loading models and tokenizer for {model_display_name}...")
    load_kwargs_hub = {"token": use_auth_token} if use_auth_token else {}
    if subfolder:
        load_kwargs_hub["subfolder"] = subfolder
        
    orpheus_model = None # Initialize to ensure they exist for del block
    tokenizer = None
    snac_model = None
        
    try:
        config_load_path = repo_id
        config_load_args = load_kwargs_hub.copy() 
        
        config = AutoConfig.from_pretrained(config_load_path, **config_load_args)
        if not hasattr(config, 'model_type'):
             raise ValueError(f"Config for {model_display_name} does not have a 'model_type' attribute.")

        tokenizer = AutoTokenizer.from_pretrained(repo_id, **load_kwargs_hub)
        if tokenizer.pad_token_id is None:
             tokenizer.add_special_tokens({'pad_token': str(PAD_TOKEN_ID)})
             tokenizer.pad_token_id = PAD_TOKEN_ID
        elif tokenizer.pad_token_id != PAD_TOKEN_ID:
            print(f"    Warning: Tokenizer pad_token_id ({tokenizer.pad_token_id}) differs from config ({PAD_TOKEN_ID}). Using {PAD_TOKEN_ID}.")
            tokenizer.pad_token_id = PAD_TOKEN_ID # Overriding if different but present

        model_load_kwargs_transformers = {"torch_dtype": torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float16 if device == "cuda" else torch.float32}
        model_load_kwargs_transformers.update(load_kwargs_hub) 

        pt_version = torch.__version__
        if tuple(map(int, pt_version.split('.')[:2])) >= (2, 0) and device=="cuda" and torch.cuda.get_device_capability()[0] >= 8:
            try:
                import flash_attn
                model_load_kwargs_transformers["attn_implementation"] = "flash_attention_2"
                print("    Flash Attention 2 is available and will be used.")
            except ImportError:
                print("    Flash Attention 2 selected but 'flash_attn' package not found. Using default attention.")
                if "attn_implementation" in model_load_kwargs_transformers:
                    del model_load_kwargs_transformers["attn_implementation"] 
        
        orpheus_model = AutoModelForCausalLM.from_pretrained(repo_id, **model_load_kwargs_transformers)
        orpheus_model.to(device)
        orpheus_model.eval()

        snac_model = SNAC.from_pretrained(SNAC_MODEL_NAME)
        snac_model.to(device)
        snac_model.eval()
        print(f"  Models for {model_display_name} loaded successfully.")

    except Exception as e:
        print(f"  ERROR loading models/tokenizer for {model_display_name}: {e}")
        import traceback
        traceback.print_exc()
        print("  Common issues: missing `config.json` or `model_type` in it at the repo root (for base model) or in subfolder (for checkpoints), private repo without auth, or `flash_attn` not installed.")
        if orpheus_model: del orpheus_model
        if tokenizer: del tokenizer
        if snac_model: del snac_model
        if device == "cuda": torch.cuda.empty_cache()
        return

    all_input_data = []
    max_input_length = 0
    total_samples_to_generate = len(target_speaker_ids) * len(prompts)
    print(f"  Preparing {total_samples_to_generate} samples for batching...")

    for speaker_idx, current_speaker_id in enumerate(target_speaker_ids):
        for i, prompt_text in enumerate(prompts):
            full_prompt_text = f"{current_speaker_id}: {prompt_text}"
            try:
                input_ids_list = (
                    [start_of_human_token] +
                    tokenizer.encode(full_prompt_text, add_special_tokens=False) +
                    [end_of_human_token, start_of_ai_token, start_of_speech_token]
                )
                max_input_length = max(max_input_length, len(input_ids_list))
                all_input_data.append({
                    'input_ids_list': input_ids_list,
                    'speaker_id': current_speaker_id,
                    'prompt_text': prompt_text,
                    'prompt_idx': i + 1
                })
            except Exception as e:
                print(f"    Error tokenizing prompt '{prompt_text[:50]}...' for {current_speaker_id}: {e}. Skipping this sample.")
                continue

    if not all_input_data:
        print("  No valid input data prepared after tokenization. Exiting for this checkpoint.")
        if orpheus_model: del orpheus_model
        if tokenizer: del tokenizer
        if snac_model: del snac_model
        if device == "cuda": torch.cuda.empty_cache()
        return

    print(f"  Max input sequence length across all samples: {max_input_length}")

    for batch_start_idx in range(0, len(all_input_data), inference_batch_size):
        batch_end_idx = min(batch_start_idx + inference_batch_size, len(all_input_data))
        current_batch_samples = all_input_data[batch_start_idx:batch_end_idx]

        batch_input_ids_padded = []
        batch_attention_mask = []
        
        for sample_data in current_batch_samples:
            input_ids_list = sample_data['input_ids_list']
            padding_needed = max_input_length - len(input_ids_list)
            padded_input_ids = input_ids_list + [tokenizer.pad_token_id] * padding_needed
            attention_mask = [1] * len(input_ids_list) + [0] * padding_needed
            batch_input_ids_padded.append(padded_input_ids)
            batch_attention_mask.append(attention_mask)

        batched_input_ids = torch.tensor(batch_input_ids_padded, dtype=torch.long, device=device)
        batched_attention_mask = torch.tensor(batch_attention_mask, dtype=torch.long, device=device)

        print(f"\n  Generating SNAC codes for batch {batch_start_idx // inference_batch_size + 1}/{ (len(all_input_data) + inference_batch_size -1) // inference_batch_size } ({len(current_batch_samples)} samples)...")
        batch_start_time = time.time()

        try:
            with torch.inference_mode():
                generated_ids_batch = orpheus_model.generate(
                    input_ids=batched_input_ids,
                    attention_mask=batched_attention_mask,
                    max_new_tokens=2048, 
                    num_beams=1,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=[end_of_speech_token, end_of_ai_token],
                )
        except Exception as e:
            print(f"    Error during Orpheus generation for batch {batch_start_idx}-{batch_end_idx}: {e}")
            import traceback; traceback.print_exc()
            continue # Skip to next batch

        batch_generation_time = time.time() - batch_start_time
        print(f"    Batch generation completed in {batch_generation_time:.2f} seconds.")

        for i, sample_data in enumerate(current_batch_samples):
            current_speaker_id = sample_data['speaker_id']
            prompt_text = sample_data['prompt_text']
            prompt_idx = sample_data['prompt_idx']
            original_input_length = len(sample_data['input_ids_list']) 
            generated_sequence_raw = generated_ids_batch[i, original_input_length:].tolist() 

            snac_codes_generated = []
            for token_id in generated_sequence_raw:
                if token_id == end_of_speech_token or token_id == end_of_ai_token:
                    break
                if token_id >= AUDIO_CODE_BASE_OFFSET and token_id < (AUDIO_CODE_BASE_OFFSET + 7 * 4096):
                    snac_codes_generated.append(token_id)
            
            if not snac_codes_generated:
                print(f"      Warning: No SNAC codes generated for {current_speaker_id} (Prompt {prompt_idx}). Skipping.")
                continue
            if len(snac_codes_generated) % 7 != 0:
                snac_codes_generated = snac_codes_generated[:-(len(snac_codes_generated) % 7)]
                if not snac_codes_generated:
                    print(f"      Warning: No valid SNAC codes left after truncation for {current_speaker_id} (Prompt {prompt_idx}). Skipping.")
                    continue

            codes_l0, codes_l1, codes_l2 = deinterleave_snac_codes(snac_codes_generated, device)
            if codes_l0 is None:
                print(f"      Failed to de-interleave codes for {current_speaker_id} (Prompt {prompt_idx}). Skipping.")
                continue
            try:
                with torch.inference_mode():
                    waveform_tensor = snac_model.decode([codes_l0, codes_l1, codes_l2])
                if waveform_tensor is None or waveform_tensor.numel() == 0:
                     print(f"      Warning: SNAC decoding produced an empty waveform for {current_speaker_id} (Prompt {prompt_idx}).")
                     continue
                audio_data = waveform_tensor.squeeze().cpu().numpy()
                # Sanitize prompt text for filename: replace non-alphanumeric with underscore, limit length
                filename_safe_prompt = re.sub(r'\W+', '_', prompt_text[:40]).strip('_')
                output_filename = f"{checkpoint_name_safe}_{current_speaker_id}_prompt{prompt_idx}_{filename_safe_prompt}.wav"
                output_filepath = os.path.join(output_dir, output_filename)
                sf.write(output_filepath, audio_data, TARGET_AUDIO_SAMPLING_RATE)
                print(f"      Successfully saved audio to: {output_filepath}")
            except Exception as e:
                print(f"      Error during SNAC decoding or saving audio for {current_speaker_id} (Prompt {prompt_idx}): {e}")
                import traceback; traceback.print_exc()
                continue
    
    # Explicitly delete models and clear cache
    if orpheus_model: del orpheus_model
    if tokenizer: del tokenizer
    if snac_model: del snac_model
    if device == "cuda":
        torch.cuda.empty_cache()
        print(f"  CUDA cache cleared for {model_display_name}.")
        
    print(f"\n--- Inference for Model {model_display_name} Finished ---")


def main_run_all_inferences(model_repo_id, checkpoint_subfolders_list, prompts, speaker_ids, base_output_dir="inference_results_multi_checkpoint", device="cuda", inference_batch_size=50, use_auth_token=None, process_base_model=True):
    print(f"\n--- Starting Multi-Checkpoint Inference Orchestration ---")
    print(f"Base Model Repository ID: {model_repo_id}")
    print(f"Base Output Directory: {base_output_dir}")
    
    models_to_process_configs = []
    if process_base_model:
        print("Will attempt to process the base model from the repo root.")
        models_to_process_configs.append((model_repo_id, None)) # (repo_id, subfolder=None for base)
    else:
        print("Skipping processing of the base model from the repo root.")
        
    for cp_subfolder in checkpoint_subfolders_list:
        models_to_process_configs.append((model_repo_id, cp_subfolder)) # (repo_id, subfolder=cp_subfolder)

    if not models_to_process_configs:
        print("No model configurations to process. Exiting.")
        return

    print(f"Total Model Configurations to Process: {len(models_to_process_configs)}")
    print(f"Total Speakers: {len(speaker_ids)}")
    print(f"Total Prompts per Speaker: {len(prompts)}")
    print(f"Expected total audio files: {len(models_to_process_configs) * len(speaker_ids) * len(prompts)}")
    print(f"Using batch size: {inference_batch_size} for inference generation.")
    if use_auth_token:
        print("Using authentication token for loading models.")


    if not os.path.exists(base_output_dir):
        os.makedirs(base_output_dir, exist_ok=True)
        print(f"Created base output directory: {base_output_dir}")

    total_start_time = time.time()
    for repo_id_iter, subfolder_iter in models_to_process_configs:
        # Create a unique directory name for this specific model/checkpoint configuration
        if subfolder_iter: # It's a checkpoint subfolder
            checkpoint_output_dirname = subfolder_iter.replace("/", "_").replace("\\", "_")
        else: # It's the base model from the root of repo_id_iter
            # Use the repo_id_iter's base name for the directory, plus "_base"
            # e.g., if repo_id_iter is "user/my-model", dirname is "my-model_base"
            dir_base_name = os.path.basename(repo_id_iter).replace("/", "_").replace("\\", "_")
            checkpoint_output_dirname = f"{dir_base_name}_base"
        
        specific_output_dir = os.path.join(base_output_dir, checkpoint_output_dirname)
        
        run_inference_for_single_checkpoint(
            repo_id=repo_id_iter, 
            subfolder=subfolder_iter, 
            prompts=prompts,
            target_speaker_ids=speaker_ids,
            output_dir=specific_output_dir, # Pass the specific output dir for this checkpoint
            device=device,
            inference_batch_size=inference_batch_size,
            use_auth_token=use_auth_token
        )
    
    total_end_time = time.time()
    print(f"\n--- All Inference Tasks Completed in {total_end_time - total_start_time:.2f} seconds ---")
    print(f"Outputs are in subdirectories of: {base_output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run multi-checkpoint inference with fine-tuned Orpheus TTS models.")
    
    # These will be used if not overridden by command-line arguments
    DEFAULT_MODEL_REPO_ID = "bharathkumar1922001/3checkpoint-10speaker-aws-H100-30per"
    DEFAULT_CHECKPOINT_SUBFOLDERS = ["checkpoint-6184", "checkpoint-7730", "checkpoint-9276"]
    
    # Your specific prompts and speakers
    HINDI_PROMPTS_ARG = [
        "यार, ब्रिलियंट आइडिया! कस्टमर फीडबैक को प्रोडक्ट डेवलपमेंट साइकिल में इंटीग्रेट करना बहुत ज़रूरी है, इससे हम मार्केट रिलेवेंट बने रहेंगे, मैं तुम्हारे इस सजेशन से 100% सहमत हूँ डेफिनिटली।",
        "देख भाई, रात को हाईवे पर ड्राइव करना है, तो विज़िबिलिटी के लिए कुछ ब्राइट कलर का पहनना अच्छा आइडिया नहीं है, पर गाड़ी के अंदर के लिए एक जैकेट रख लेना यार।",
        "यार, तुम्हे क्या लगता है, क्या आर्टिफिशियल इंटेलिजेंस फ्यूचर में ह्यूमन क्रिएटिविटी को रिप्लेस कर सकता है, जैसे राइटिंग या म्यूजिक कंपोजिशन में, क्या ये पॉसिबल है?",
        "अगर तुम अपनी पब्लिक स्पीकिंग स्किल्स इम्प्रूव करना चाहते हो, तो टोस्टमास्टर्स क्लब जॉइन करना काफी फायदे मंद हो सकता है, वहां सपोर्टिव एनवायरनमेंट मिलता है प्रैक्टिस करने के लिए, भाई!",
        "वाह, क्या बात है! ये तो कमाल का सुझाव है, बॉस! मुझे पूरा यकीन है, इससे हमारे काम में बहुत सुधार आएगा।"
    ]

    TARGET_SPEAKER_IDS_ARG = [
        "aisha", "anika", "arfa", "asmr", "nikita",
        "raju", "rhea", "ruhaan", "sangeeta", "shayana"
    ]

    parser.add_argument(
        "--model_repo_id", type=str, default=DEFAULT_MODEL_REPO_ID,
        help="The base Hugging Face model repository ID from which checkpoints are subfolders (or the model itself if --skip_base_model is not used with --checkpoint_subfolders)."
    )
    parser.add_argument(
        "--checkpoint_subfolders", nargs='*', default=DEFAULT_CHECKPOINT_SUBFOLDERS,
        help="List of checkpoint subfolder names within the model_repo_id. If empty, only the base model (root of model_repo_id) will be processed unless --skip_base_model is also set."
    )
    parser.add_argument(
        "--skip_base_model", action='store_true',
        help="If set, skips trying to load and process the model from the root of model_repo_id. Only processes specified --checkpoint_subfolders."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="inference_results_multi_checkpoint",
        help="Base directory to save the generated .wav files. Subdirectories will be created for each model/checkpoint."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
        help="Device to run inference on ('cuda' or 'cpu')."
    )
    parser.add_argument(
        "--inference_batch_size",
        type=int,
        default=50, # You might need to adjust this based on H100 VRAM for a 3B model
        help="Number of samples to process in a single batch during model generation. Adjust based on VRAM."
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="Optional Hugging Face token if accessing private models. Can also be set via environment variable HUGGING_FACE_HUB_TOKEN or `huggingface-cli login`."
    )

    args = parser.parse_args()
    auth_token = args.hf_token if args.hf_token else os.environ.get("HUGGING_FACE_HUB_TOKEN")

    # Determine if the base model should be processed
    # If --skip_base_model is True, then process_base_model is False.
    # If --skip_base_model is False (default), then process_base_model is True.
    should_process_base = not args.skip_base_model

    main_run_all_inferences(
        model_repo_id=args.model_repo_id,
        checkpoint_subfolders_list=args.checkpoint_subfolders, # Pass the list of subfolders
        prompts=HINDI_PROMPTS_ARG,                              # Use the hardcoded prompts
        speaker_ids=TARGET_SPEAKER_IDS_ARG,                     # Use the hardcoded speakers
        base_output_dir=args.output_dir,
        device=args.device,
        inference_batch_size=args.inference_batch_size,
        use_auth_token=auth_token,
        process_base_model=should_process_base
    )