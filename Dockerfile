# Install python and its packages

FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

# TF fails to find "libcudart.so.10.1"
# First, theres /usr/local/nvidia/lib64 in LD_LIBRARY_PATH, but /usr/local/nvidia does not exist
# We link the existing /usr/local/cuda-10.2 to fill that role.
# RUN ln -s /usr/local/cuda-10.2 /usr/local/nvidia \
# 	&& ln -s /usr/local/cuda-10.2/lib64/libcudart.so.10.2 /usr/local/nvidia/lib64/libcudart.so.10.1

# Update package lists and install Python and pip
# Install build tools and libraries
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential python pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Test if Python and pip are working
#RUN python3 --version && pip3 --version

# scientific
RUN pip install \
	numpy scipy scikit-learn \
	matplotlib seaborn \
	pillow beautifulsoup4 fire \
	pandas \
	langdetect openai \
    nltk PyYAML pysbd \
    textdistance \
    transformers

RUN pip install --no-deps genalog==0.1.0
# interactive
# RUN pip install \
#	ipython jupyterlab tqdm

# pytorch
# use shared cache for pytorch
ENV TORCH_MODEL_ZOO /dhlab1/pytorch_model_zoo/models
ENV TORCH_HOME /dhlab1/pytorch_model_zoo
# install pytorch for cuda11
#
#RUN pip --no-cache-dir install \
#	torch torchvision torchaudio \
#	--extra-index-url https://download.pytorch.org/whl/cu113 \
#	&& pip --no-cache-dir install tensorboard \
#	&& pip --no-cache-dir install --no-deps kornia

# Update pip and setuptools
#RUN python -m pip install --upgrade pip setuptools

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
#RUN python -m pip install --upgrade pip
#RUN pip install -r requirements.txt

# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variable
ENV NAME World

# Ensure run_parallel.sh is executable
RUN chmod +x run_one.sh

# Run run_parallel.sh when the container launches
CMD ["./run_one.sh"]