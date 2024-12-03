# Use an official Python runtime as a parent image
FROM python:3.8.5


# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0

# Install pip and Python dependencies
RUN pip install --upgrade pip==20.3
RUN pip install \
    torch==1.13.0+cu116 torchvision==0.12.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html \
    numpy==1.19.2 \
    albumentations==0.4.3 \
    diffusers \
    opencv-python==4.1.2.30 \
    pudb==2019.2 \
    invisible-watermark \
    imageio==2.9.0 \
    imageio-ffmpeg==0.4.2 \
    pytorch-lightning==1.4.2 \
    omegaconf==2.1.1 \
    test-tube>=0.7.5 \
    streamlit>=0.73.1 \
    einops==0.3.0 \
    torch-fidelity==0.3.0 \
    transformers==4.19.2 \
    torchmetrics==0.6.0 \
    kornia==0.6 \
    git+https://github.com/CompVis/taming-transformers.git@master#egg=taming-transformers \
    git+https://github.com/openai/CLIP.git@main#egg=clip \
    -e .

