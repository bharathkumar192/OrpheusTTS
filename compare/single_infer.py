# --- START OF FILE single_infer.py ---
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
SNAC_MODEL_NAME = "hubertsiuzdak/snac_24khz"
TARGET_AUDIO_SAMPLING_RATE = 24000
tokeniser_length = 128256
start_of_speech_token = tokeniser_length + 1
end_of_speech_token = tokeniser_length + 2
start_of_human_token = tokeniser_length + 3
end_of_human_token = tokeniser_length + 4
start_of_ai_token = tokeniser_length + 5
end_of_ai_token = tokeniser_length + 6
AUDIO_CODE_BASE_OFFSET = tokeniser_length + 10
PAD_TOKEN_ID = 128263

SNAC_OFFSETS = [
    AUDIO_CODE_BASE_OFFSET + 0 * 4096, AUDIO_CODE_BASE_OFFSET + 1 * 4096,
    AUDIO_CODE_BASE_OFFSET + 2 * 4096, AUDIO_CODE_BASE_OFFSET + 3 * 4096,
    AUDIO_CODE_BASE_OFFSET + 4 * 4096, AUDIO_CODE_BASE_OFFSET + 5 * 4096,
    AUDIO_CODE_BASE_OFFSET + 6 * 4096,
]

def deinterleave_snac_codes(snac_codes_flat, device):
    if not snac_codes_flat or len(snac_codes_flat) % 7 != 0:
        print(f"Warning: Invalid flat SNAC code list length ({len(snac_codes_flat)}). Must be multiple of 7 or non-empty.")
        return None, None, None
    num_frames = len(snac_codes_flat) // 7
    codes_lvl0, codes_lvl1, codes_lvl2 = [], [], []
    for i in range(0, len(snac_codes_flat), 7):
        try:
            if not (SNAC_OFFSETS[0] <= snac_codes_flat[i] < SNAC_OFFSETS[0] + 4096): raise ValueError(f"L0 code {snac_codes_flat[i]} out of range for expected offset {SNAC_OFFSETS[0]}")
            if not (SNAC_OFFSETS[1] <= snac_codes_flat[i+1] < SNAC_OFFSETS[1] + 4096): raise ValueError(f"L1a code {snac_codes_flat[i+1]} out of range for expected offset {SNAC_OFFSETS[1]}")
            if not (SNAC_OFFSETS[2] <= snac_codes_flat[i+2] < SNAC_OFFSETS[2] + 4096): raise ValueError(f"L2a code {snac_codes_flat[i+2]} out of range for expected offset {SNAC_OFFSETS[2]}")
            if not (SNAC_OFFSETS[3] <= snac_codes_flat[i+3] < SNAC_OFFSETS[3] + 4096): raise ValueError(f"L2b code {snac_codes_flat[i+3]} out of range for expected offset {SNAC_OFFSETS[3]}")
            if not (SNAC_OFFSETS[4] <= snac_codes_flat[i+4] < SNAC_OFFSETS[4] + 4096): raise ValueError(f"L1b code {snac_codes_flat[i+4]} out of range for expected offset {SNAC_OFFSETS[4]}")
            if not (SNAC_OFFSETS[5] <= snac_codes_flat[i+5] < SNAC_OFFSETS[5] + 4096): raise ValueError(f"L2c code {snac_codes_flat[i+5]} out of range for expected offset {SNAC_OFFSETS[5]}")
            if not (SNAC_OFFSETS[6] <= snac_codes_flat[i+6] < SNAC_OFFSETS[6] + 4096): raise ValueError(f"L2d code {snac_codes_flat[i+6]} out of range for expected offset {SNAC_OFFSETS[6]}")
            codes_lvl0.append(snac_codes_flat[i] - SNAC_OFFSETS[0])
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
        codes_lvl0_t = torch.tensor(codes_lvl0, dtype=torch.long, device=device).unsqueeze(0)
        codes_lvl1_t = torch.tensor(codes_lvl1, dtype=torch.long, device=device).unsqueeze(0)
        codes_lvl2_t = torch.tensor(codes_lvl2, dtype=torch.long, device=device).unsqueeze(0)
        if codes_lvl0_t.shape[1]!=num_frames or codes_lvl1_t.shape[1]!=2*num_frames or codes_lvl2_t.shape[1]!=4*num_frames:
            raise ValueError("Tensor shape mismatch after de-interleaving.")
        return codes_lvl0_t, codes_lvl1_t, codes_lvl2_t
    except Exception as e:
        print(f"Error creating tensors during de-interleaving: {e}")
        return None, None, None

def run_inference_for_single_checkpoint(repo_id, subfolder, prompts, target_speaker_ids, output_dir, device="cuda", use_auth_token=None):
    if subfolder:
        checkpoint_name_safe = subfolder.replace("/", "_").replace("\\", "_")
        model_display_name = f"{repo_id}/{subfolder}"
    else:
        checkpoint_name_safe = os.path.basename(repo_id).replace("/", "_").replace("\\", "_") + "_base"
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

    orpheus_model, tokenizer_obj, snac_model_obj = None, None, None
    try:
        config = AutoConfig.from_pretrained(repo_id, **load_kwargs_hub)
        if not hasattr(config, 'model_type'):
             raise ValueError(f"Config for {model_display_name} does not have 'model_type'.")

        # Addressing the padding warning: set padding_side to 'left' for decoder-only models
        tokenizer_obj = AutoTokenizer.from_pretrained(repo_id, padding_side='left', **load_kwargs_hub)
        if tokenizer_obj.pad_token_id is None:
             tokenizer_obj.add_special_tokens({'pad_token': str(PAD_TOKEN_ID)})
             tokenizer_obj.pad_token_id = PAD_TOKEN_ID
        elif tokenizer_obj.pad_token_id != PAD_TOKEN_ID:
            print(f"    Warning: Tokenizer pad_token_id ({tokenizer_obj.pad_token_id}) differs. Using {PAD_TOKEN_ID}.")
            tokenizer_obj.pad_token_id = PAD_TOKEN_ID
        
        # Ensure pad token is also set for the model config if generating
        if config.pad_token_id is None or config.pad_token_id != tokenizer_obj.pad_token_id:
            config.pad_token_id = tokenizer_obj.pad_token_id


        model_load_kwargs_transformers = {"torch_dtype": torch.bfloat16 if device=="cuda" and torch.cuda.is_bf16_supported() else torch.float16 if device=="cuda" else torch.float32}
        model_load_kwargs_transformers.update(load_kwargs_hub)
        if tuple(map(int, torch.__version__.split('.')[:2])) >= (2,0) and device=="cuda" and torch.cuda.get_device_capability()[0] >= 8:
            try:
                import flash_attn
                model_load_kwargs_transformers["attn_implementation"] = "flash_attention_2"
                print("    Flash Attention 2 is available and will be used.")
            except ImportError:
                print("    Flash Attention 2 selected but 'flash_attn' not found.")
        
        orpheus_model = AutoModelForCausalLM.from_pretrained(repo_id, config=config, **model_load_kwargs_transformers)
        orpheus_model.to(device)
        orpheus_model.eval()

        snac_model_obj = SNAC.from_pretrained(SNAC_MODEL_NAME)
        snac_model_obj.to(device)
        snac_model_obj.eval()
        print(f"  Models for {model_display_name} loaded successfully.")
    except Exception as e:
        print(f"  ERROR loading models/tokenizer for {model_display_name}: {e}")
        import traceback; traceback.print_exc()
        return # Exit this checkpoint processing

    total_prompts_for_checkpoint = len(target_speaker_ids) * len(prompts)
    generated_count = 0

    for speaker_idx, current_speaker_id in enumerate(target_speaker_ids):
        for prompt_idx_loop, prompt_text in enumerate(prompts):
            current_prompt_num = prompt_idx_loop + 1
            print(f"\n  Processing Speaker: {current_speaker_id} ({speaker_idx+1}/{len(target_speaker_ids)}), Prompt: {current_prompt_num}/{len(prompts)}")
            print(f"    Text: '{prompt_text[:60]}...'")
            single_gen_start_time = time.time()

            full_prompt_text = f"{current_speaker_id}: {prompt_text}"
            try:
                # For single inference, tokenizer handles padding if needed by model.generate with left padding.
                # The prompt itself is not padded before tokenization.
                input_ids_list = (
                    [start_of_human_token] +
                    tokenizer_obj.encode(full_prompt_text, add_special_tokens=False) +
                    [end_of_human_token, start_of_ai_token, start_of_speech_token]
                )
                input_ids = torch.tensor([input_ids_list], device=device)
            except Exception as e:
                print(f"    Error tokenizing prompt: {e}. Skipping.")
                continue

            try:
                with torch.inference_mode():
                    generated_ids = orpheus_model.generate(
                        input_ids,
                        max_new_tokens=2048,
                        num_beams=1,
                        do_sample=False,
                        pad_token_id=tokenizer_obj.pad_token_id, # Crucial for generation
                        eos_token_id=[end_of_speech_token, end_of_ai_token],
                    )
                
                generated_sequence_raw = generated_ids[0, input_ids.shape[1]:].tolist()
                snac_codes_generated = []
                for token_id in generated_sequence_raw:
                    if token_id == end_of_speech_token or token_id == end_of_ai_token: break
                    if AUDIO_CODE_BASE_OFFSET <= token_id < (AUDIO_CODE_BASE_OFFSET + 7 * 4096):
                        snac_codes_generated.append(token_id)

                if not snac_codes_generated:
                    print(f"    Warning: No SNAC codes generated. Skipping.")
                    continue
                if len(snac_codes_generated) % 7 != 0:
                    snac_codes_generated = snac_codes_generated[:-(len(snac_codes_generated) % 7)]
                    if not snac_codes_generated:
                        print(f"    Warning: No valid SNAC codes after truncation. Skipping.")
                        continue
                
                # print(f"    Generated {len(snac_codes_generated)} SNAC codes raw: {snac_codes_generated[:14]}...") # For debugging

                codes_l0, codes_l1, codes_l2 = deinterleave_snac_codes(snac_codes_generated, device)
                if codes_l0 is None:
                    print(f"    Failed to de-interleave codes. Skipping.")
                    continue

                with torch.inference_mode():
                    waveform_tensor = snac_model_obj.decode([codes_l0, codes_l1, codes_l2])
                if waveform_tensor is None or waveform_tensor.numel() == 0:
                    print(f"    Warning: SNAC decoding produced empty waveform. Skipping.")
                    continue

                audio_data = waveform_tensor.squeeze().cpu().numpy()
                filename_safe_prompt = re.sub(r'\W+', '_', prompt_text[:40]).strip('_')
                output_filename = f"{checkpoint_name_safe}_{current_speaker_id}_prompt{current_prompt_num}_{filename_safe_prompt}.wav"
                output_filepath = os.path.join(output_dir, output_filename)
                sf.write(output_filepath, audio_data, TARGET_AUDIO_SAMPLING_RATE)
                print(f"    Successfully saved: {output_filepath} ({(time.time() - single_gen_start_time):.2f}s)")
                generated_count +=1

            except Exception as e:
                print(f"    Error during generation/decoding for this prompt: {e}")
                import traceback; traceback.print_exc()
                continue
    
    print(f"  Generated {generated_count}/{total_prompts_for_checkpoint} audio files for this checkpoint.")
    if orpheus_model: del orpheus_model
    if tokenizer_obj: del tokenizer_obj
    if snac_model_obj: del snac_model_obj
    if device == "cuda":
        torch.cuda.empty_cache()
        print(f"  CUDA cache cleared for {model_display_name}.")
    print(f"--- Inference for Model {model_display_name} Finished ---")

def main_run_all_inferences(model_repo_id, checkpoint_subfolders_list, prompts, speaker_ids, base_output_dir="single_inference_results", device="cuda", use_auth_token=None, process_base_model=True):
    print(f"\n--- Starting Single-Prompt Multi-Checkpoint Inference ---")
    print(f"Base Model Repository ID: {model_repo_id}")
    print(f"Base Output Directory: {base_output_dir}")

    models_to_process = []
    if process_base_model:
        models_to_process.append((model_repo_id, None))
    for cp_subfolder in checkpoint_subfolders_list:
        models_to_process.append((model_repo_id, cp_subfolder))

    if not models_to_process:
        print("No models to process. Exiting."); return

    print(f"Total Model Configs: {len(models_to_process)}, Speakers: {len(speaker_ids)}, Prompts/Speaker: {len(prompts)}")
    print(f"Expected total audio files: {len(models_to_process) * len(speaker_ids) * len(prompts)}")
    if use_auth_token: print("Using auth token.")

    if not os.path.exists(base_output_dir):
        os.makedirs(base_output_dir, exist_ok=True)

    total_start_time = time.time()
    for repo_id_iter, subfolder_iter in models_to_process:
        dir_name_part = subfolder_iter.replace("/","_") if subfolder_iter else os.path.basename(repo_id_iter).replace("/","_") + "_base"
        specific_out_dir = os.path.join(base_output_dir, dir_name_part)
        run_inference_for_single_checkpoint(
            repo_id_iter, subfolder_iter, prompts, speaker_ids, specific_out_dir,
            device, use_auth_token
        )
    total_time_taken = time.time() - total_start_time
    print(f"\n--- All Inference Tasks Completed in {total_time_taken:.2f} seconds ---")
    print(f"Outputs are in subdirectories of: {base_output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run single-prompt multi-checkpoint inference.")
    DEFAULT_MODEL_REPO_ID = "bharathkumar1922001/3checkpoint-10speaker-aws-H100-30per"
    DEFAULT_CHECKPOINT_SUBFOLDERS = ["checkpoint-6184", "checkpoint-7730", "checkpoint-9276"]
    HINDI_PROMPTS_ARG = [
        "यार, ब्रिलियंट आइडिया! कस्टमर फीडबैक को प्रोडक्ट डेवलपमेंट साइकिल में इंटीग्रेट करना बहुत ज़रूरी है, इससे हम मार्केट रिलेवेंट बने रहेंगे, मैं तुम्हारे इस सजेशन से 100% सहमत हूँ डेफिनिटली।",
        "देख भाई, रात को हाईवे पर ड्राइव करना है, तो विज़िबिलिटी के लिए कुछ ब्राइट कलर का पहनना अच्छा आइडिया नहीं है, पर गाड़ी के अंदर के लिए एक जैकेट रख लेना यार।",
        "यार, तुम्हे क्या लगता है, क्या आर्टिफिशियल इंटेलिजेंस फ्यूचर में ह्यूमन क्रिएटिविटी को रिप्लेस कर सकता है, जैसे राइटिंग या म्यूजिक कंपोजिशन में, क्या ये पॉसिबल है?",
        "अगर तुम अपनी पब्लिक स्पीकिंग स्किल्स इम्प्रूव करना चाहते हो, तो टोस्टमास्टर्स क्लब जॉइन करना काफी फायदे मंद हो सकता है, वहां सपोर्टिव एनवायरनमेंट मिलता है प्रैक्टिस करने के लिए, भाई!",
        "वाह, क्या बात है! ये तो कमाल का सुझाव है, बॉस! मुझे पूरा यकीन है, इससे हमारे काम में बहुत सुधार आएगा।"
    ]
    TARGET_SPEAKER_IDS_ARG = [
        "aisha", "anika", "arfa", "asmr", "nikita", "raju", "rhea", "ruhaan", "sangeeta", "shayana"
    ]

    parser.add_argument("--model_repo_id", type=str, default=DEFAULT_MODEL_REPO_ID)
    parser.add_argument("--checkpoint_subfolders", nargs='*', default=DEFAULT_CHECKPOINT_SUBFOLDERS)
    parser.add_argument("--skip_base_model", action='store_true')
    parser.add_argument("--output_dir", type=str, default="single_inference_output")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--hf_token", type=str, default=None)
    args = parser.parse_args()

    auth_token = args.hf_token or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    should_process_base = not args.skip_base_model

    main_run_all_inferences(
        args.model_repo_id, args.checkpoint_subfolders, HINDI_PROMPTS_ARG, TARGET_SPEAKER_IDS_ARG,
        args.output_dir, args.device, auth_token, should_process_base
    )
# --- END OF FILE single_infer.py ---