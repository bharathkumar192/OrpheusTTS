import os
import torch
import torchaudio
import torchaudio.transforms as T
from transformers import AutoModelForCausalLM, AutoTokenizer
from snac import SNAC 
import soundfile as sf
import numpy as np
import argparse
import time

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

# --- Helper Function ---
def deinterleave_snac_codes(snac_codes_flat, device):
    """
    De-interleaves the flat list of 7 SNAC codes per frame back into
    the 3 separate code levels expected by the SNAC decoder.
    """
    if not snac_codes_flat or len(snac_codes_flat) % 7 != 0:
        print(f"Warning: Invalid flat SNAC code list length ({len(snac_codes_flat)}). Must be multiple of 7.")
        return None, None, None

    num_frames = len(snac_codes_flat) // 7
    codes_lvl0 = []
    codes_lvl1 = []
    codes_lvl2 = []

    for i in range(0, len(snac_codes_flat), 7):
        # Subtract the base offset for each codebook to get the actual code index (0-4095)
        codes_lvl0.append(snac_codes_flat[i]   - SNAC_OFFSETS[0]) # L0
        codes_lvl1.append(snac_codes_flat[i+1] - SNAC_OFFSETS[1]) # L1a
        codes_lvl2.append(snac_codes_flat[i+2] - SNAC_OFFSETS[2]) # L2a
        codes_lvl2.append(snac_codes_flat[i+3] - SNAC_OFFSETS[3]) # L2b
        codes_lvl1.append(snac_codes_flat[i+4] - SNAC_OFFSETS[4]) # L1b
        codes_lvl2.append(snac_codes_flat[i+5] - SNAC_OFFSETS[5]) # L2c
        codes_lvl2.append(snac_codes_flat[i+6] - SNAC_OFFSETS[6]) # L2d

    # Convert to tensors and reshape
    # Shape: [1, T_frames] for L0
    # Shape: [1, 2 * T_frames] for L1
    # Shape: [1, 4 * T_frames] for L2
    try:
        codes_lvl0_tensor = torch.tensor(codes_lvl0, dtype=torch.long, device=device).unsqueeze(0)
        codes_lvl1_tensor = torch.tensor(codes_lvl1, dtype=torch.long, device=device).unsqueeze(0)
        codes_lvl2_tensor = torch.tensor(codes_lvl2, dtype=torch.long, device=device).unsqueeze(0)

        # Sanity check shapes
        if codes_lvl0_tensor.shape[1] != num_frames: raise ValueError("L0 shape mismatch")
        if codes_lvl1_tensor.shape[1] != 2 * num_frames: raise ValueError("L1 shape mismatch")
        if codes_lvl2_tensor.shape[1] != 4 * num_frames: raise ValueError("L2 shape mismatch")

    except Exception as e:
        print(f"Error creating tensors during de-interleaving: {e}")
        return None, None, None

    return codes_lvl0_tensor, codes_lvl1_tensor, codes_lvl2_tensor

# --- Main Inference Function ---
def run_inference(model_checkpoint_path, prompts, target_speaker_id, output_dir="inference_output", device="cuda"):
    """
    Runs inference using the fine-tuned Orpheus model and SNAC decoder.

    Args:
        model_checkpoint_path (str): Path to the fine-tuned Orpheus checkpoint directory.
        prompts (list[str]): List of text sentences to generate audio for.
        target_speaker_id (str): The speaker ID to use (must match training).
        output_dir (str): Directory to save the generated .wav files.
        device (str): "cuda" or "cpu".
    """
    print(f"--- Starting Inference ---")
    print(f"Model Checkpoint: {model_checkpoint_path}")
    print(f"Speaker ID: {target_speaker_id}")
    print(f"Output Directory: {output_dir}")
    print(f"Using device: {device}")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created output directory: {output_dir}")

    # --- Load Models and Tokenizer ---
    print("\n--- Loading Models and Tokenizer ---")
    try:
        print(f"Loading Orpheus tokenizer from: {model_checkpoint_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint_path)
        # Ensure pad token is set correctly (important for generation)
        if tokenizer.pad_token_id is None:
             print(f"Tokenizer pad_token_id is None. Setting to default: {PAD_TOKEN_ID}")
             tokenizer.add_special_tokens({'pad_token': str(PAD_TOKEN_ID)}) # Should already be in vocab if trained correctly
             tokenizer.pad_token_id = PAD_TOKEN_ID

        print(f"Loading Orpheus model from: {model_checkpoint_path}")
        # Load with appropriate dtype for inference (matching training if possible)
        model_load_kwargs = {"torch_dtype": torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float16 if device == "cuda" else torch.float32}
        # Attempt to use Flash Attention 2 if available and on CUDA
        pt_version = torch.__version__
        if tuple(map(int, pt_version.split('.')[:2])) >= (2, 0) and device=="cuda" and torch.cuda.get_device_capability()[0] >= 8:
            model_load_kwargs["attn_implementation"] = "flash_attention_2"
            print("Attempting to use Flash Attention 2 for model loading.")
        
        orpheus_model = AutoModelForCausalLM.from_pretrained(model_checkpoint_path, **model_load_kwargs)
        orpheus_model.to(device)
        orpheus_model.eval()
        print("Orpheus model loaded and set to eval mode.")

        print(f"Loading SNAC model ({SNAC_MODEL_NAME}) for decoding...")
        snac_model = SNAC.from_pretrained(SNAC_MODEL_NAME)
        snac_model.to(device)
        snac_model.eval()
        print("SNAC model loaded and set to eval mode.")

    except Exception as e:
        print(f"ERROR loading models/tokenizer: {e}")
        import traceback
        traceback.print_exc()
        return

    # --- Generate Audio for Each Prompt ---
    for i, prompt_text in enumerate(prompts):
        print(f"\n--- Processing Prompt {i+1}/{len(prompts)} ---")
        print(f"Text: '{prompt_text}'")

        start_time = time.time()

        # 1. Format Input
        full_prompt_text = f"{target_speaker_id}: {prompt_text}"
        try:
            input_ids_list = (
                [start_of_human_token] +
                tokenizer.encode(full_prompt_text, add_special_tokens=False) +
                [end_of_human_token, start_of_ai_token, start_of_speech_token] # Model generates from <SOS>
            )
            input_ids = torch.tensor([input_ids_list], device=device)
            print(f"Input tokens length: {input_ids.shape[1]}")
        except Exception as e:
            print(f"Error tokenizing prompt: {e}")
            continue

        # 2. Generate SNAC Codes with Orpheus
        print("Generating SNAC codes with Orpheus model...")
        try:
            with torch.inference_mode():
                generated_ids = orpheus_model.generate(
                    input_ids,
                    max_new_tokens=2048, # Max SNAC tokens (7 per frame, ~32ms/frame). 2048 = ~292 frames = ~9.3 sec. Adjust if needed.
                    num_beams=1,         # Greedy decoding
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=[end_of_speech_token, end_of_ai_token], # Stop at either EOSpeech or EOA
                    # temperature=0.7, # Keep deterministic for audio codes usually
                    # top_k=50,        # Keep deterministic for audio codes usually
                )
            gen_duration = time.time() - start_time
            print(f"Orpheus generation took {gen_duration:.2f} seconds.")

            # Extract generated part
            generated_sequence = generated_ids[0][input_ids.shape[1]:].tolist()
            print(f"Generated {len(generated_sequence)} new tokens.")

            # 3. Extract SNAC codes from generated sequence
            snac_codes_generated = []
            for token_id in generated_sequence:
                if token_id == end_of_speech_token or token_id == end_of_ai_token:
                    break # Stop collecting codes
                # Basic check: is it likely an audio code?
                if token_id >= AUDIO_CODE_BASE_OFFSET and token_id < (AUDIO_CODE_BASE_OFFSET + 4096 * 7):
                    snac_codes_generated.append(token_id)

            print(f"Extracted {len(snac_codes_generated)} SNAC code tokens.")

            if not snac_codes_generated:
                print("Warning: No SNAC codes were generated or extracted. Skipping audio decoding.")
                continue
            if len(snac_codes_generated) % 7 != 0:
                print(f"Warning: Number of extracted SNAC codes ({len(snac_codes_generated)}) is not a multiple of 7. Truncating...")
                snac_codes_generated = snac_codes_generated[:-(len(snac_codes_generated) % 7)]
                if not snac_codes_generated:
                    print("Warning: No valid SNAC codes left after truncation. Skipping.")
                    continue
                print(f"Using {len(snac_codes_generated)} SNAC codes after truncation.")


        except Exception as e:
            print(f"Error during Orpheus generation: {e}")
            import traceback
            traceback.print_exc()
            continue

        # 4. De-interleave SNAC codes
        print("De-interleaving SNAC codes...")
        try:
            codes_l0, codes_l1, codes_l2 = deinterleave_snac_codes(snac_codes_generated, device)
            if codes_l0 is None:
                print("Failed to de-interleave codes. Skipping decoding.")
                continue
            print(f"De-interleaved shapes: L0={codes_l0.shape}, L1={codes_l1.shape}, L2={codes_l2.shape}")
        except Exception as e:
            print(f"Error during de-interleaving: {e}")
            continue

        # 5. Decode SNAC codes to Audio using SNAC Model
        print("Decoding SNAC codes to waveform using SNAC model...")
        decode_start_time = time.time()
        try:
            with torch.inference_mode():
                # The SNAC model expects a list of the 3 code tensors
                waveform_tensor = snac_model.decode([codes_l0, codes_l1, codes_l2])
            decode_duration = time.time() - decode_start_time
            print(f"SNAC decoding took {decode_duration:.2f} seconds.")

            if waveform_tensor is None or waveform_tensor.numel() == 0:
                 print("Warning: SNAC decoding produced an empty waveform.")
                 continue

            # 6. Save Audio
            audio_data = waveform_tensor.squeeze().cpu().numpy() # Shape [T_audio]
            filename_safe_prompt = "".join(c if c.isalnum() else "_" for c in prompt_text[:40])
            output_filename = f"output_{target_speaker_id}_{i+1}_{filename_safe_prompt}.wav"
            output_filepath = os.path.join(output_dir, output_filename)

            sf.write(output_filepath, audio_data, TARGET_AUDIO_SAMPLING_RATE)
            total_time = time.time() - start_time
            print(f"Successfully saved audio to: {output_filepath}")
            print(f"Total time for prompt: {total_time:.2f} seconds")

        except Exception as e:
            print(f"Error during SNAC decoding or saving audio: {e}")
            import traceback
            traceback.print_exc()
            continue

    print("\n--- Inference Script Finished ---")


# --- Argument Parser ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference with a fine-tuned Orpheus TTS model.")
    parser.add_argument(
        "--model_checkpoint",
        type=str,
        required=True,
        help="Path to the fine-tuned Orpheus model checkpoint directory (e.g., './checkpoints_aisha_H200_good_run_v1/')."
    )
    parser.add_argument(
        "--prompts",
        nargs='+', # Allows multiple prompts separated by spaces
        required=True,
        help="List of text prompts to generate audio for (e.g., 'नमस्ते' 'आप कैसे हैं?')."
    )
    parser.add_argument(
        "--speaker_id",
        type=str,
        required=True,
        help="The target speaker ID used during training (e.g., 'aisha')."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="inference_output",
        help="Directory to save the generated .wav files."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
        help="Device to run inference on ('cuda' or 'cpu')."
    )

    args = parser.parse_args()

    # Basic validation
    if not os.path.isdir(args.model_checkpoint):
        print(f"Error: Model checkpoint directory not found: {args.model_checkpoint}")
        exit(1)
    if not args.prompts:
        print(f"Error: No prompts provided.")
        exit(1)


    run_inference(
        model_checkpoint_path=args.model_checkpoint,
        prompts=args.prompts,
        target_speaker_id=args.speaker_id,
        output_dir=args.output_dir,
        device=args.device
    )
