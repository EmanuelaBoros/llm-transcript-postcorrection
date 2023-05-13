# Install python and its packages
FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

# Update package lists and install Python, pip, and software-properties-common
# Install build tools and libraries
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends build-essential software-properties-common wget && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.11 python3-pip python3.11-venv && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*


# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

# Set path to conda
ENV PATH /opt/conda/bin:$PATH

RUN echo $PATH

RUN ls /opt/conda/bin

RUN which conda
# Creating Python environment
RUN conda create -n py311 python=3.11

# Activate environment and install packages
RUN echo "source activate py311" > ~/.bashrc
#ENV PATH /opt/conda/envs/py311/bin:$PATH

# Install Python libraries
RUN python -m pip install --upgrade pip setuptools && \
    pip install \
	numpy scipy scikit-learn \
	matplotlib seaborn \
	pillow beautifulsoup4 fire \
	pandas multidict \
	langdetect openai \
    nltk PyYAML pysbd \
    textdistance \
    transformers \
    --no-deps genalog==0.1.0



# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variable
ENV NAME World

# Ensure run_parallel.sh is executable
RUN chmod +x run_one.sh

# Run run_parallel.sh when the container launches
CMD ["./run_one.sh", "impresso", "prompt_basic_01.txt", "data/config_cluster.yaml"]
