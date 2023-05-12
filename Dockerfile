# Install python and its packages
FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

# Update package lists and install Python, pip, and software-properties-common
# Install build tools and libraries
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends build-essential software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.11 python3-pip python3.11-venv && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create a virtual environment and activate it
RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python libraries
RUN python -m pip install --upgrade pip setuptools && \
    pip install \
	numpy scipy scikit-learn \
	matplotlib seaborn \
	pillow beautifulsoup4 fire \
	pandas \
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
