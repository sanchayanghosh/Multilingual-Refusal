# Use a valid, existing PyTorch CUDA runtime tag
FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/root/.cache/huggingface

# System dependencies for git, build tools, and OpenCV/plotting backends
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Install core Python dependencies for Gemma-2-9B-IT, HF checkpointing, and analysis
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    transformers \
    accelerate \
    datasets \
    huggingface_hub \
    safetensors \
    jaxtyping \
    mmengine \
    jsonpickle \
    lm-eval \
    scikit-learn \
    matplotlib \
    seaborn \
    numpy \
    scipy \
    tqdm \
    pyyaml

# If your repo has a requirements.txt, copy and install it cleanly
COPY requirements.txt* ./
RUN if [ -f requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; fi

# Copy the repository code into /workspace
COPY . /workspace

CMD ["/bin/bash"]
