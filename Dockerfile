# Install python and its packages
FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

ENV LIBRARY_PATH=/usr/local/cuda/lib64/stubs
SHELL ["/bin/bash", "-cu"]
WORKDIR /
ENV USER_NAME=eboros
ENV HOME=/home/eboros
ENV CONDA_PREFIX=/home/eboros/.conda
ENV CONDA=/home/eboros/.conda/condabin/conda

#RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub && \
#    apt-get update && \
#    apt-get install -y --allow-downgrades --allow-change-held-packages --no-install-recommends build-essential cmake g++-4.8 git curl vim unzip wget tmux screen ca-certificates apt-utils libjpeg-dev libpng-dev && \
#    rm -rf /var/lib/apt/lists/*
# Update package lists and install Python, pip, and software-properties-common
# Install build tools and libraries
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends build-essential \
    git curl vim unzip wget tmux screen ca-certificates apt-utils software-properties-common wget && \
    apt-get update && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /home/eboros
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p ${CONDA_PREFIX} && \
    rm miniconda.sh && \
    ${CONDA} config --set auto_activate_base false && \
    ${CONDA} init bash && \
    ${CONDA} create --name myenv python=3.11

WORKDIR /home/eboros/app

RUN /home/eboros/.conda/condabin/conda create -n myenv python=3.11 pip
RUN /home/eboros/.conda/condabin/conda run -n myenv pip install --upgrade pip setuptools
RUN /home/eboros/.conda/condabin/conda run -n myenv pip install \
	numpy scipy scikit-learn \
	matplotlib seaborn \
	pillow beautifulsoup4 fire \
	pandas multidict \
	langdetect openai \
    nltk PyYAML pysbd \
    textdistance jsonlines \
    torch transformers
RUN /home/eboros/.conda/condabin/conda run -n myenv pip install genalog==0.1.0 --no-deps


COPY . /home/eboros/app

RUN ls -la
USER eboros
WORKDIR /home/eboros/app

# Ensure run_parallel.sh is executable
RUN chmod +x /home/eboros/app/run_one.sh

# Run run_parallel.sh when the container launches
CMD ["./run_one.sh", "impresso", "prompt_basic_01.txt", "data/config_cluster.yml"]
