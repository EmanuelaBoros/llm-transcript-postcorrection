# Set base image
FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

# Set environment variables for user
ENV USER_NAME=eboros
ENV USER_ID=268532
ENV GROUP_NAME=DHLAB-unit
ENV GROUP_ID=11703

# Install sudo
#RUN apt-get update && apt-get install -y sudo

# Install build tools and libraries
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends build-essential \
    git curl vim unzip wget tmux screen ca-certificates apt-utils software-properties-common wget && \
    apt-get install -y sudo && \
    apt-get update && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create a group and user
RUN groupadd -g $GROUP_ID $GROUP_NAME
RUN useradd -ms /bin/bash -u $USER_ID -g $GROUP_ID $USER_NAME

# Add new user to sudoers
RUN echo "${USER_NAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Add Conda
ENV CONDA_PREFIX=/home/eboros/.conda
ENV CONDA=/home/eboros/.conda/condabin/conda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p ${CONDA_PREFIX} && \
    rm miniconda.sh && \
    ${CONDA} config --set auto_activate_base false && \
    ${CONDA} init bash && \
    ${CONDA} create --name myenv python=3.11

ENV PATH="/home/eboros/.conda/envs/myenv/bin:$PATH"

RUN /home/eboros/.conda/condabin/conda create -n myenv python=3.11 pip

RUN /home/eboros/.conda/condabin/conda run -n myenv pip install --upgrade pip setuptools
RUN /home/eboros/.conda/condabin/conda run -n myenv pip install \
	numpy scipy scikit-learn \
	matplotlib seaborn \
	pillow beautifulsoup4 fire \
	pandas multidict sentencepiece \
	langdetect openai fairscale \
    nltk PyYAML pysbd \
    textdistance jsonlines \
    torch transformers sentencepiece
RUN /home/eboros/.conda/condabin/conda run -n myenv pip install genalog==0.1.0 --no-deps


ENV PATH="/home/eboros/.conda/envs/myenv/bin:$PATH"

# Set the working directory
WORKDIR /home/$USER_NAME/app

# Copy app directory
COPY . .

# Change ownership of the copied files to the new user and group
RUN chown -R $USER_NAME:$GROUP_NAME /home/$USER_NAME/app

# Switch to the new user
USER $USER_NAME

# Make sure your script is executable
RUN chmod +x run_one.sh

# Run run_parallel.sh when the container launches
#CMD ["./run_one.sh", "impresso", "prompt_basic_01.txt", "data/config_cluster.yml"]
CMD ["sleep", "infinity"]