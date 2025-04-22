FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

RUN apt update && apt install -y git wget espeak-ng ffmpeg libsndfile1-dev libespeak-ng-dev build-essential python3 python3-pip python-is-python3

RUN pip install --upgrade pip
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
RUN pip install transformers accelerate datasets huggingface_hub librosa phonemizer wandb einops matplotlib

RUN ln -s /usr/bin/espeak-ng /usr/bin/espeak

WORKDIR /workspace