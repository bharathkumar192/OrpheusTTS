import modal

app = modal.App("test-flash-attn-image-app")

cuda_version = "12.8.0"  # should be no greater than host CUDA version
flavor = "devel"  #  includes full CUDA toolkit
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.11")
    .apt_install("git")
    .pip_install(  # required to build flash-attn
        "ninja",
        "packaging",
        "wheel",
        "torch", # Add torchaudio as well, as your main app uses it
        "torchaudio" # It's good practice if torch is used with audio
    )
    .pip_install(  # add flash-attn
        "flash-attn==2.7.4.post1", extra_options="--no-build-isolation"
    )
)

@app.function(gpu="a10g", image=image, timeout=1800) # Increased timeout for build
def run_flash_attn_test():
    print("Attempting to import torch and flash_attn...")
    import torch
    from flash_attn import flash_attn_func
    print("Imports successful!")

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")

    batch_size, seqlen, nheads, headdim, nheads_k = 2, 4, 3, 16, 3

    q = torch.randn(batch_size, seqlen, nheads, headdim, dtype=torch.float16).to("cuda")
    k = torch.randn(batch_size, seqlen, nheads_k, headdim, dtype=torch.float16).to("cuda")
    v = torch.randn(batch_size, seqlen, nheads_k, headdim, dtype=torch.float16).to("cuda")
    print("Test tensors created on CUDA.")

    out = flash_attn_func(q, k, v)
    assert out.shape == (batch_size, seqlen, nheads, headdim)
    print("flash_attn_func executed successfully!")
    return "Flash Attention test passed!"

@app.local_entrypoint()
def main():
    print("Running flash_attn test function on Modal...")
    result = run_flash_attn_test.remote()
    print(f"Test result: {result}") 