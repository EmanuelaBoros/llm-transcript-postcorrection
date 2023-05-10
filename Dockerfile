# Use an official Python runtime as a parent image
FROM nvidia/cuda:12.1.1-base-ubi8

# Install Python 3.8 and other required packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends python3.11 python3-pip python3.11-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Update pip and setuptools
RUN python3.8 -m pip install --upgrade pip setuptools

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt

# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variable
ENV NAME World

# Ensure run_parallel.sh is executable
RUN chmod +x run_parallel.sh

# Run run_parallel.sh when the container launches
CMD ["./run_parallel.sh"]
