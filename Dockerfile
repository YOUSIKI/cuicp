FROM nvcr.io/nvidia/rapidsai/rapidsai-core:22.10-cuda11.5-base-ubuntu20.04-py3.9

RUN pip install --upgrade --no-cache-dir \
    tqdm

WORKDIR /workspace
VOLUME [ "/workspace" ]
