import modal
import os
import io
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download
from snac import SNAC # Assuming the library is imported as 'snac' after installing 'snac-codec'
import soundfile as sf
import numpy as np
import time
import fastapi
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field, validator
from typing import Optional
import traceback # For detailed error logging

# --- Configuration Constants ---
MODEL_ID = "bharathkumar1922001/10speaker-aws-9ksteps-7epochs-8B2G"
ORPHEUS_CACHE_PATH = "/model_cache/orpheus"
SNAC_CACHE_PATH = "/model_cache/snac"

AVAILABLE_SPEAKERS = ["aisha", "anika", "arfa", "asmr", "nikita", "raju", "rhea", "ruhaan", "sangeeta", "shayana"]
DEFAULT_SPEAKER = "shayana"

SNAC_MODEL_NAME = "hubertsiuzdak/snac_24khz"
TARGET_AUDIO_SAMPLING_RATE = 24000

TOKENISER_LENGTH = 128256
START_OF_SPEECH_TOKEN = TOKENISER_LENGTH + 1
END_OF_SPEECH_TOKEN = TOKENISER_LENGTH + 2
START_OF_HUMAN_TOKEN = TOKENISER_LENGTH + 3
END_OF_HUMAN_TOKEN = TOKENISER_LENGTH + 4
START_OF_AI_TOKEN = TOKENISER_LENGTH + 5
END_OF_AI_TOKEN = TOKENISER_LENGTH + 6
AUDIO_CODE_BASE_OFFSET = TOKENISER_LENGTH + 10
PAD_TOKEN_ID_EXPECTED = 128263

SNAC_OFFSETS = [AUDIO_CODE_BASE_OFFSET + i * 4096 for i in range(7)]

_GPU_TYPE_ENV = "H100"
if "H100" in _GPU_TYPE_ENV.upper():
    GPU_CONFIG = _GPU_TYPE_ENV
    TORCH_CUDA_ARCH_LIST_FOR_BUILD = "9.0"
elif "A100" in _GPU_TYPE_ENV.upper():
    GPU_CONFIG = _GPU_TYPE_ENV
    TORCH_CUDA_ARCH_LIST_FOR_BUILD = "8.0"
elif "A10G" in _GPU_TYPE_ENV.upper():
    GPU_CONFIG = _GPU_TYPE_ENV
    TORCH_CUDA_ARCH_LIST_FOR_BUILD = "8.6"
else:
    GPU_CONFIG = "A10G"
    TORCH_CUDA_ARCH_LIST_FOR_BUILD = "8.6"
print(f"Using GPU: {GPU_CONFIG}, Target CUDA Arch for Flash-Attn build: {TORCH_CUDA_ARCH_LIST_FOR_BUILD}")

CONTAINER_TIMEOUT = int(os.environ.get("CONTAINER_TIMEOUT", "150")) # Increased to 2.5 mins

class TTSRequest(BaseModel):
    text: str = Field(..., description="The text to convert to speech")
    speaker_id: str = Field(default=DEFAULT_SPEAKER, description="The speaker voice ID to use")
    max_new_tokens: int = Field(default=2048, ge=100, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    repetition_penalty: float = Field(default=1.1, ge=1.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    output_sample_rate: int = Field(default=TARGET_AUDIO_SAMPLING_RATE)
    audio_quality_preset: str = Field(default="high", pattern="^(low|medium|high)$")
    speed_adjustment: float = Field(default=1.0, ge=0.5, le=2.0)
    seed: Optional[int] = Field(default=None)
    early_stopping: bool = Field(default=True)

    class Config:
        json_schema_extra = {"example": {"text": "नमस्ते", "speaker_id": "shayana", "temperature": 0.7}}

    @validator('speaker_id')
    def validate_speaker(cls, v):
        if v.lower() not in AVAILABLE_SPEAKERS:
            raise ValueError(f"Speaker '{v}' not available. Choose from: {', '.join(AVAILABLE_SPEAKERS)}")
        return v.lower()

AUDIO_QUALITY_PRESETS = {
    "low": {"target_sr": 16000, "subtype": "PCM_16"},
    "medium": {"target_sr": 24000, "subtype": "PCM_16"},
    "high": {"target_sr": TARGET_AUDIO_SAMPLING_RATE, "subtype": "PCM_16"},
}

hf_secret = modal.Secret.from_name("huggingface-secrets")

def download_all_models():
    hf_token_for_download = os.environ.get("HF_TOKEN")
    os.makedirs(ORPHEUS_CACHE_PATH, exist_ok=True)
    print(f"Downloading Orpheus model ({MODEL_ID}) to {ORPHEUS_CACHE_PATH}...")
    snapshot_download(
        repo_id=MODEL_ID, local_dir=ORPHEUS_CACHE_PATH,
        local_dir_use_symlinks=False, token=hf_token_for_download
    )
    print("Orpheus model downloaded.")
    os.makedirs(SNAC_CACHE_PATH, exist_ok=True)
    print(f"Downloading SNAC model ({SNAC_MODEL_NAME}) to {SNAC_CACHE_PATH}...")
    snapshot_download(
        repo_id=SNAC_MODEL_NAME, local_dir=SNAC_CACHE_PATH,
        local_dir_use_symlinks=False, token=hf_token_for_download
    )
    print("SNAC model downloaded.")

cuda_image_tag = "12.4.1-devel-ubuntu22.04"
orpheus_image = (
    modal.Image.from_registry(f"nvidia/cuda:{cuda_image_tag}", add_python="3.11")
    .apt_install("git")
    .pip_install(["ninja", "packaging", "wheel", "torch==2.3.1", "torchaudio==2.3.1"])
    .env({
        "TORCH_CUDA_ARCH_LIST": TORCH_CUDA_ARCH_LIST_FOR_BUILD,
        "MAX_JOBS": str(os.cpu_count() or 4),
    })
    .run_commands("pip install flash-attn==2.6.3 --no-build-isolation || echo 'Flash attention install failed, will fallback to default attention'")
    .pip_install([
        "transformers==4.46.1", "huggingface_hub", "hf_transfer", "accelerate==0.30.1",
        "snac", "soundfile", "numpy", "scipy", "hf_xet",
        "fastapi", "uvicorn[standard]", "pydantic",
    ])
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .run_function(
        download_all_models, secrets=[hf_secret], timeout=3600
    )
)

app = modal.App("orpheus-tts-api-10speakers-v2-fa2fix")

@app.cls(
    gpu=GPU_CONFIG, image=orpheus_image,
    scaledown_window=CONTAINER_TIMEOUT, secrets=[hf_secret], max_containers=3,
)
class OrpheusTTSAPI:
    def __init__(self):
        self.web_app = fastapi.FastAPI(
            title="Orpheus TTS API - 10 Speakers (FA2 Fix Attempt)",
            description="TTS using Orpheus and SNAC. Attempting robust Flash Attention 2.",
            version="2.1.1"
        )
        self.web_app.add_middleware(
            CORSMiddleware, allow_origins=["*"], allow_credentials=True,
            allow_methods=["*"], allow_headers=["*"],
        )
        self._add_api_routes()

    def _add_api_routes(self):
        @self.web_app.post("/generate", response_class=Response)
        async def http_generate_speech(request: TTSRequest):
            request_received_time = time.time()
            print(f"Received /generate request at {request_received_time:.3f}")
            try:
                wav_bytes, duration = self._perform_tts(request)
                response_sent_time = time.time()
                print(f"TTS generation completed in {duration:.3f}s (internal). Total request handling time: {response_sent_time - request_received_time:.3f}s")
                return Response(content=wav_bytes, media_type="audio/wav")
            except Exception as e:
                print(f"Error in /generate endpoint: {e}")
                traceback.print_exc()
                return Response(
                    content=f'{{"error": "TTS generation failed: {str(e)}"}}',
                    media_type="application/json", status_code=500
                )
        # ... (other endpoints: /, /health, /speakers - unchanged) ...
        @self.web_app.get("/")
        async def root():
            return {
                "message": "Orpheus TTS API - 10 Speakers (FA2 Fix Attempt) is running.",
                "model_id": MODEL_ID,
                "gpu_config": GPU_CONFIG,
                "flash_attention_explicitly_loaded": self.flash_attn_explicitly_loaded,
                "actual_attn_implementation": self.actual_attn_implementation,
                "model_dtype": str(self.orpheus_model.dtype) if hasattr(self, 'orpheus_model') else "N/A",
                "available_speakers": AVAILABLE_SPEAKERS,
                "docs_url": "/docs"
            }

        @self.web_app.get("/health")
        async def health_check():
            return {"status": "healthy", "timestamp": time.time()}

        @self.web_app.get("/speakers")
        async def get_speakers():
            return {
                "available_speakers": AVAILABLE_SPEAKERS,
                "default_speaker": DEFAULT_SPEAKER,
                "total_speakers": len(AVAILABLE_SPEAKERS),
                "speaker_descriptions": {
                    "aisha": "Girlfriend female speaker - warm and intimate voice",
                    "anika": "Social female speaker - energetic and friendly voice",
                    "arfa": "Professional female speaker - confident and clear voice",
                    "asmr": "ASMR style speaker - soft and calming whisper voice",
                    "nikita": "Youthful female speaker - bright and cheerful voice",
                    "raju": "Relatable male speaker - casual and approachable voice",
                    "rhea": "Late-night girlfriend speaker - sultry and soothing voice",
                    "ruhaan": "Mature male speaker - deep and authoritative voice",
                    "sangeeta": "Warm female speaker - gentle and nurturing voice",
                    "shayana": "Customer care female speaker - polite and helpful voice (default)"
                }
            }

    @modal.enter()
    def load_all_models_and_tokenizer(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"--- Initializing models on device: {self.device} ---")
        self.flash_attn_explicitly_loaded = False # Flag to track if FA2 was loaded
        self.actual_attn_implementation = "default"

        # --- Path Checks (Unchanged) ---
        # ... (your path checks for ORPHEUS_CACHE_PATH and SNAC_CACHE_PATH) ...
        print(f"Checking existence of ORPHEUS_CACHE_PATH: {ORPHEUS_CACHE_PATH}")
        if os.path.exists(ORPHEUS_CACHE_PATH) and os.path.isdir(ORPHEUS_CACHE_PATH):
            print(f"Contents of {ORPHEUS_CACHE_PATH}: {os.listdir(ORPHEUS_CACHE_PATH)}")
        else:
            print(f"CRITICAL ERROR: {ORPHEUS_CACHE_PATH} does not exist or is not a directory!")
        print(f"Checking existence of SNAC_CACHE_PATH: {SNAC_CACHE_PATH}")
        if os.path.exists(SNAC_CACHE_PATH) and os.path.isdir(SNAC_CACHE_PATH):
            print(f"Contents of {SNAC_CACHE_PATH}: {os.listdir(SNAC_CACHE_PATH)}")
        else:
            print(f"CRITICAL ERROR: {SNAC_CACHE_PATH} does not exist or is not a directory!")


        # --- Load Tokenizer (Unchanged) ---
        print(f"Loading Orpheus tokenizer from cached path: {ORPHEUS_CACHE_PATH}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            ORPHEUS_CACHE_PATH, trust_remote_code=True, local_files_only=True
        )
        # ... (your pad token check) ...
        if self.tokenizer.pad_token_id is None:
            print(f"Warning: Tokenizer pad_token_id is None. Setting to expected: {PAD_TOKEN_ID_EXPECTED}")
            self.tokenizer.add_special_tokens({'pad_token': str(PAD_TOKEN_ID_EXPECTED)})
            self.tokenizer.pad_token_id = PAD_TOKEN_ID_EXPECTED
        elif self.tokenizer.pad_token_id != PAD_TOKEN_ID_EXPECTED:
             print(f"Warning: Loaded tokenizer pad_token_id ({self.tokenizer.pad_token_id}) "
                   f"differs from expected ({PAD_TOKEN_ID_EXPECTED}). Using loaded tokenizer's pad_token_id.")
        else:
            print(f"Tokenizer pad_token_id ({self.tokenizer.pad_token_id}) is correctly set.")


        # --- Load Orpheus Model with FA2 focus ---
        print(f"Attempting to load Orpheus model ({MODEL_ID}) from {ORPHEUS_CACHE_PATH}")
        
        model_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        print(f"Target model dtype: {model_dtype}")

        # Attempt 1: Load directly to GPU with FA2 specified
        model_load_args_fa2 = {
            "trust_remote_code": True,
            "local_files_only": True,
            "torch_dtype": model_dtype,
            "attn_implementation": "flash_attention_2",
            # "device_map": self.device # This can sometimes interfere with FA2 if not careful, try without first
        }
        
        try:
            print(f"Attempt 1: Loading Orpheus model with explicit FA2 and device='{self.device}'...")
            # For FA2 with from_pretrained, model should ideally be on GPU *during* initialization
            # or transformers needs to handle the deferred FA2 module initialization correctly.
            # Loading to CPU then moving to GPU (`.to(self.device)`) is a common pattern,
            # but for FA2, ensuring it "knows" it's for GPU from the start can be more robust.
            # However, directly using `device_map` with `attn_implementation` can be tricky.
            # A safer bet is to load and then immediately move.
            
            self.orpheus_model = AutoModelForCausalLM.from_pretrained(ORPHEUS_CACHE_PATH, **model_load_args_fa2)
            self.orpheus_model.to(self.device) # Ensure it's on device AFTER loading with FA2 hint

            # Verification
            if hasattr(self.orpheus_model.config, "_attn_implementation") and \
               self.orpheus_model.config._attn_implementation == "flash_attention_2":
                self.flash_attn_explicitly_loaded = True
                self.actual_attn_implementation = "flash_attention_2"
                print("Orpheus model loaded successfully WITH Flash Attention 2 (verified by config).")
            else:
                # Heuristic check if FA2 modules exist
                fa2_module_found = False
                for _, module in self.orpheus_model.named_modules():
                    if "FlashAttention" in module.__class__.__name__ or "FlashSelfAttention" in module.__class__.__name__:
                        fa2_module_found = True
                        break
                if fa2_module_found:
                    self.flash_attn_explicitly_loaded = True
                    self.actual_attn_implementation = "flash_attention_2 (heuristic)"
                    print("Orpheus model loaded, Flash Attention 2 modules detected (heuristic).")
                else:
                    print("Warning: Orpheus model loaded, FA2 requested, but not confirmed by config or module check. May be using default attention.")
                    self.actual_attn_implementation = "default (fallback or unconfirmed FA2)"

        except Exception as e_fa2_attempt:
            print(f"Error during Attempt 1 (explicit FA2 load): {e_fa2_attempt}")
            print(traceback.format_exc())
            print("Falling back to loading model without explicit FA2 hint, then trying to apply if possible or using default.")
            
            # Fallback: Load without FA2 hint, then check
            model_load_args_default = {
                "trust_remote_code": True, "local_files_only": True, "torch_dtype": model_dtype,
            }
            try:
                self.orpheus_model = AutoModelForCausalLM.from_pretrained(ORPHEUS_CACHE_PATH, **model_load_args_default)
                self.orpheus_model.to(self.device)
                self.actual_attn_implementation = "default (initial fallback load)"
                print("Orpheus model loaded with default attention (fallback).")
                # You could potentially try model.to_bettertransformer() here if applicable,
                # or other post-load optimizations if FA2 direct load fails.
            except Exception as e_fallback_load:
                print(f"CRITICAL: Failed to load Orpheus model even with fallback: {e_fallback_load}")
                raise

        # Final diagnostic log after loading attempts
        print(f"--- Orpheus Model Post-Load Diagnostics ---")
        print(f"  Device: {self.orpheus_model.device if hasattr(self, 'orpheus_model') else 'N/A'}")
        print(f"  Dtype: {self.orpheus_model.dtype if hasattr(self, 'orpheus_model') else 'N/A'}")
        print(f"  Flash Attention Explicitly Loaded (flag): {self.flash_attn_explicitly_loaded}")
        print(f"  Actual Attention Implementation: {self.actual_attn_implementation}")
        if hasattr(self.orpheus_model, 'config') and hasattr(self.orpheus_model.config, '_attn_implementation'):
             print(f"  Model Config _attn_implementation: {self.orpheus_model.config._attn_implementation}")
        else:
             print(f"  Model Config _attn_implementation: Not set or found.")
        print(f"--- End Orpheus Model Post-Load Diagnostics ---")


        # Check vocab size compatibility (Unchanged)
        model_vocab_size = self.orpheus_model.get_input_embeddings().weight.shape[0]
        tokenizer_vocab_size = len(self.tokenizer)
        if tokenizer_vocab_size != model_vocab_size:
            print(f"CRITICAL MISMATCH: Tokenizer vocab ({tokenizer_vocab_size}) vs Model vocab ({model_vocab_size})!")

        self.orpheus_model.eval() # Ensure eval mode
        print("Orpheus model setup on device and in eval mode complete.")

        # --- Load SNAC Model (Unchanged) ---
        print(f"Loading SNAC model ({SNAC_MODEL_NAME}) from {SNAC_CACHE_PATH}...")
        # ... (SNAC loading code - unchanged) ...
        try:
            self.snac_model = SNAC.from_pretrained(SNAC_CACHE_PATH)
        except Exception as e:
            print(f"Failed to load SNAC from {SNAC_CACHE_PATH} directly ({e}). "
                  f"Attempting to load {SNAC_MODEL_NAME} (may trigger download if not in SNAC's internal cache).")
            self.snac_model = SNAC.from_pretrained(SNAC_MODEL_NAME)
        self.snac_model.to(self.device); self.snac_model.eval()
        print("SNAC model loaded.")
        print("--- All models loaded successfully. ---")


    def _deinterleave_snac_codes(self, snac_codes_flat: list[int]) -> tuple[Optional[torch.Tensor], ...]:
        # ... (unchanged) ...
        if not snac_codes_flat or len(snac_codes_flat) % 7 != 0: return None, None, None
        num_frames = len(snac_codes_flat) // 7
        codes_lvl0, codes_lvl1, codes_lvl2 = ([] for _ in range(3))
        for i in range(0, len(snac_codes_flat), 7):
            codes_lvl0.append(snac_codes_flat[i]   - SNAC_OFFSETS[0])
            codes_lvl1.append(snac_codes_flat[i+1] - SNAC_OFFSETS[1]) 
            codes_lvl2.append(snac_codes_flat[i+2] - SNAC_OFFSETS[2]) 
            codes_lvl2.append(snac_codes_flat[i+3] - SNAC_OFFSETS[3]) 
            codes_lvl1.append(snac_codes_flat[i+4] - SNAC_OFFSETS[4]) 
            codes_lvl2.append(snac_codes_flat[i+5] - SNAC_OFFSETS[5]) 
            codes_lvl2.append(snac_codes_flat[i+6] - SNAC_OFFSETS[6]) 
        try:
            t_kwargs = {"dtype": torch.long, "device": self.device}
            tensors = [torch.tensor(l, **t_kwargs).unsqueeze(0) for l in [codes_lvl0, codes_lvl1, codes_lvl2]]
            if not (tensors[0].shape[1] == num_frames and \
                    tensors[1].shape[1] == 2 * num_frames and \
                    tensors[2].shape[1] == 4 * num_frames):
                # Add more detailed logging for shape mismatch
                print(f"Shape mismatch details: num_frames={num_frames}")
                print(f"  L0 expected shape: (1, {num_frames}), got: {tensors[0].shape}")
                print(f"  L1 expected shape: (1, {2*num_frames}), got: {tensors[1].shape}")
                print(f"  L2 expected shape: (1, {4*num_frames}), got: {tensors[2].shape}")
                raise ValueError("Shape mismatch after tensor creation in deinterleave.")
            return tuple(tensors)
        except Exception as e:
            print(f"Error in _deinterleave_snac_codes: {e}"); traceback.print_exc(); return None, None, None

    def _adjust_audio_speed(self, waveform_tensor: torch.Tensor, speed_factor: float) -> torch.Tensor:
        # ... (unchanged) ...
        if abs(speed_factor - 1.0) < 1e-3: return waveform_tensor
        original_sr = TARGET_AUDIO_SAMPLING_RATE
        if waveform_tensor.ndim == 1: waveform_tensor = waveform_tensor.unsqueeze(0)
        if waveform_tensor.shape[0] != 1: waveform_tensor = torch.mean(waveform_tensor, dim=0, keepdim=True)
        try:
            import torchaudio.functional as AF
            # Ensure waveform is on CPU for torchaudio.functional.speed if it has issues with CUDA tensors for this op
            resampled_waveform, _ = AF.speed(waveform_tensor.squeeze(0).cpu(), sample_rate=original_sr, factor=speed_factor)
            return resampled_waveform.unsqueeze(0).to(self.device) # Move back to original device
        except Exception as e:
            print(f"Torchaudio speed adjustment failed ({e}), falling back to basic interpolation (less robust).")
            # ... (rest of your fallback interpolation logic - unchanged) ...
            orig_length = waveform_tensor.shape[1]
            new_length = int(orig_length / speed_factor)
            if new_length == 0: return torch.empty((1,0), device=self.device)
            indices = torch.linspace(0, orig_length - 1, new_length, device=self.device)
            indices_floor = torch.clamp(indices.long(), 0, orig_length - 1)
            indices_ceil = torch.clamp(indices_floor + 1, 0, orig_length - 1)
            weight_ceil = indices - indices_floor.float(); weight_floor = 1.0 - weight_ceil
            adjusted_waveform = waveform_tensor[:, indices_floor] * weight_floor + \
                                waveform_tensor[:, indices_ceil] * weight_ceil
            return adjusted_waveform


    def _perform_tts(self, request: TTSRequest) -> tuple[bytes, float]:
        # Add the detailed profiling from previous suggestion here
        # ... (unchanged from your previous app.py, but now with FA2 hopefully active) ...
        # --- Start of _perform_tts ---
        request_start_time_tts = time.time()
        current_call_times = {"tokenization": 0.0, "llm_gen": 0.0, "snac_extraction": 0.0, "snac_decode": 0.0, "post_processing": 0.0}

        if request.seed is not None: torch.manual_seed(request.seed); np.random.seed(request.seed)

        tokenization_start_time = time.time()
        full_prompt = f"{request.speaker_id}: {request.text}"
        try:
            input_ids = [START_OF_HUMAN_TOKEN] + \
                        self.tokenizer.encode(full_prompt, add_special_tokens=False) + \
                        [END_OF_HUMAN_TOKEN, START_OF_AI_TOKEN, START_OF_SPEECH_TOKEN]
            input_ids_t = torch.tensor([input_ids], device=self.device)
        except Exception as e: raise RuntimeError(f"Tokenization failed: {e}")
        current_call_times["tokenization"] = time.time() - tokenization_start_time
        
        llm_gen_start_time = time.time()
        try:
            with torch.inference_mode(): # Crucial for performance and correct behavior
                generation_params = {
                    "input_ids": input_ids_t, "max_new_tokens": request.max_new_tokens, "num_beams": 1,
                    "do_sample": request.temperature > 0.001 and request.temperature <=2.0, # Allow temp up to 2.0 for sampling
                    "repetition_penalty": request.repetition_penalty,
                    "pad_token_id": self.tokenizer.pad_token_id,
                    "eos_token_id": [END_OF_SPEECH_TOKEN, END_OF_AI_TOKEN],
                    "early_stopping": request.early_stopping
                }
                if generation_params["do_sample"]:
                    generation_params["temperature"] = max(0.01, request.temperature)
                    generation_params["top_p"] = request.top_p
                else: # Not sampling, remove sampling-specific keys
                    generation_params.pop("temperature", None)
                    generation_params.pop("top_p", None)
                
                print(f"  Orpheus generation params: {generation_params}")
                generated_ids = self.orpheus_model.generate(**generation_params)
        except Exception as e: 
            print(f"Orpheus model generation failed: {e}"); traceback.print_exc(); raise
        current_call_times["llm_gen"] = time.time() - llm_gen_start_time
        print(f"  LLM token generation took: {current_call_times['llm_gen']:.3f}s")

        snac_extraction_start_time = time.time()
        generated_part = generated_ids[0][input_ids_t.shape[1]:].tolist()
        snac_codes = []
        for token_val in generated_part:
            if token_val == END_OF_SPEECH_TOKEN or token_val == END_OF_AI_TOKEN: break
            if AUDIO_CODE_BASE_OFFSET <= token_val < (AUDIO_CODE_BASE_OFFSET + 4096 * 7):
                snac_codes.append(token_val)
        
        print(f"  Number of SNAC codes generated: {len(snac_codes)}")
        if not snac_codes: raise RuntimeError("No SNAC codes generated.")
        if len(snac_codes) % 7 != 0:
            print(f"  SNAC codes not multiple of 7 (len={len(snac_codes)}). Truncating.")
            snac_codes = snac_codes[:-(len(snac_codes) % 7)]
        if not snac_codes: raise RuntimeError("No valid SNAC codes after truncation.")
        current_call_times["snac_extraction"] = time.time() - snac_extraction_start_time

        snac_decode_start_time = time.time()
        codes_l0, codes_l1, codes_l2 = self._deinterleave_snac_codes(snac_codes)
        if codes_l0 is None: raise RuntimeError("Failed to deinterleave SNAC codes.")
        try:
            with torch.inference_mode(): # Crucial
                waveform_t = self.snac_model.decode([codes_l0, codes_l1, codes_l2])
        except Exception as e:
            print(f"SNAC decode failed: {e}"); traceback.print_exc(); raise
        current_call_times["snac_decode"] = time.time() - snac_decode_start_time
        print(f"  SNAC deinterleave & decode took: {current_call_times['snac_decode']:.3f}s")

        if waveform_t is None or waveform_t.numel() == 0: raise RuntimeError("SNAC decoding resulted in an empty waveform.")

        post_processing_start_time = time.time()
        if abs(request.speed_adjustment - 1.0) > 1e-3:
            waveform_t = self._adjust_audio_speed(waveform_t, request.speed_adjustment)

        audio_np = waveform_t.squeeze().cpu().numpy()
        quality_cfg = AUDIO_QUALITY_PRESETS.get(request.audio_quality_preset, AUDIO_QUALITY_PRESETS["high"])
        final_sr_to_use = request.output_sample_rate
        current_audio_sr = TARGET_AUDIO_SAMPLING_RATE

        if final_sr_to_use != current_audio_sr:
            print(f"Resampling audio from {current_audio_sr}Hz to {final_sr_to_use}Hz.")
            try:
                import torchaudio.transforms as AT
                # Ensure audio_np is on the correct device for resampling if needed by AT.Resample
                temp_tensor_for_resample = torch.from_numpy(audio_np).to(self.device).unsqueeze(0) # Add batch and channel if mono
                resampler = AT.Resample(orig_freq=current_audio_sr, new_freq=final_sr_to_use).to(self.device)
                resampled_audio_tensor = resampler(temp_tensor_for_resample)
                audio_np = resampled_audio_tensor.squeeze().cpu().numpy()
            except Exception as resample_e:
                print(f"Warning: Could not resample audio with torchaudio: {resample_e}. Outputting at {current_audio_sr}Hz.")
                final_sr_to_use = current_audio_sr # Fallback to current SR

        wav_buffer = io.BytesIO()
        sf.write(wav_buffer, audio_np, final_sr_to_use, format='WAV', subtype=quality_cfg["subtype"])
        current_call_times["post_processing"] = time.time() - post_processing_start_time
        print(f"  Audio post-processing took: {current_call_times['post_processing']:.3f}s")

        total_tts_duration = time.time() - request_start_time_tts
        print(f"  Total _perform_tts duration: {total_tts_duration:.3f}s. Breakdown: {current_call_times}")
        return wav_buffer.getvalue(), total_tts_duration
        # --- End of _perform_tts ---

    @modal.exit()
    def cleanup(self):
        # ... (unchanged) ...
        print(f"{self.__class__.__name__} (GPU: {GPU_CONFIG}) container shutting down.")
        if hasattr(self, 'orpheus_model'): del self.orpheus_model
        if hasattr(self, 'snac_model'): del self.snac_model
        if hasattr(self, 'tokenizer'): del self.tokenizer
        if torch.cuda.is_available(): torch.cuda.empty_cache()


    @modal.asgi_app()
    def run_fastapi_app(self): return self.web_app

@app.local_entrypoint()
def cli_test_generation(
    prompt: str = "नमस्ते, आपका दिन कैसा रहा?",
    speaker: str = DEFAULT_SPEAKER,
    output_file: str = "cli_test_fa2fix_output.wav"
):
    # ... (unchanged) ...
    output_dir_local = "modal_test_outputs_fa2fix" 
    os.makedirs(output_dir_local, exist_ok=True)

    print(f"--- Running CLI Test on Modal for {app.name} ---")
    print(f"Prompt: '{prompt}', Speaker: '{speaker}'")

    request_data = TTSRequest(text=prompt, speaker_id=speaker)

    try:
        service_instance = OrpheusTTSAPI()
        print("Conceptual CLI test: Simulating call to remote _perform_tts.")
        print("To test deployed app, hit the HTTP /generate endpoint.")
        print("If running with `modal run app.py --prompt ...`, this test runs on Modal infra.")

        # This will execute on Modal if you run `modal run app.py --prompt "..."`
        audio_bytes, duration_secs = OrpheusTTSAPI()._perform_tts.remote(request_data)
        
        full_output_path = os.path.join(output_dir_local, output_file)
        with open(full_output_path, "wb") as f:
            f.write(audio_bytes)
        
        print(f"SUCCESS: CLI test audio saved to local file '{full_output_path}'")
        print(f"Generation duration on Modal: {duration_secs:.2f}s")

    except Exception as e:
        print(f"CLI Test FAILED: {e}")
        traceback.print_exc()