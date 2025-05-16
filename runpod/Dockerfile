# Dockerfile for Network Volume model loading & direct registry push

# Base Image (ensure this matches your needs for CUDA/PyTorch)
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
# ENV HF_HUB_ENABLE_HF_TRANSFER=1 # Not strictly needed for build if models are on volume
                                # but good if app_runpod.py ever downloads anything else (unlikely now)

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ninja-build \
    curl \
    # git-lfs is NOT needed now as models are on the volume
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt from your local context (or also fetch if it's in Git)
# Assuming requirements.txt is in the same directory when you run 'docker build'
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Install flash-attn
RUN MAX_JOBS=4 pip install flash-attn --no-build-isolation

# Fetch app_runpod.py from GitHub
# Ensure the URL is the raw content URL
ADD https://raw.githubusercontent.com/bharathkumar192/orpheus-docker/refs/heads/main/app_runpod.py app_runpod.py

# If 'snac' is a local module/directory and NOT a pip package, you'd need to get it into the image.
# If it's in the same GitHub repo, you could use ADD for that too, or include it in your build context.
# For now, assuming 'snac' is pip-installable via requirements.txt or already in the base image.
# Example if snac was also in that Git repo:
# ADD https://raw.githubusercontent.com/bharathkumar192/orpheus-docker/refs/heads/main/snac.py snac.py # If it's a single file
# Or if it's a directory, you'd typically clone the repo or part of it, or include 'snac' in your local Docker build context and use COPY.
# COPY snac ./snac # If 'snac' is a local directory next to your Dockerfile

# Models are NOT downloaded or copied during build. They are on the Network Volume.

# CMD to start the Runpod handler script
CMD ["python", "-u", "app_runpod.py"]