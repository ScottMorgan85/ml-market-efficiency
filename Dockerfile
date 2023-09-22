FROM python:3.9-slim-buster

ARG NB_USER="sagemaker-user"
ARG NB_UID="1000"
ARG NB_GID="100"

# Create a group with GID 100 (if not exists) and a user named "sagemaker-user" with UID 1000
RUN \
    apt-get update && \
    apt-get install -y sudo && \
    getent group ${NB_GID} || groupadd -g ${NB_GID} ${NB_USER} && \
    useradd -m -s /bin/bash -N -u $NB_UID -g $NB_GID $NB_USER && \
    chmod g+w /etc/passwd && \
    # Allow sagemaker-user to have passwordless sudo capability (this may be removed if not necessary)
    echo "${NB_USER}    ALL=(ALL)    NOPASSWD:    ALL" >> /etc/sudoers && \
    # Prevent apt-get cache from being persisted to this layer
    rm -rf /var/lib/apt/lists/*

# Set the working directory and copy the requirements
WORKDIR /app
COPY requirements.txt .

# Install Python packages and ipykernel
RUN pip install --no-cache-dir -r requirements.txt ipykernel && \
    python -m ipykernel install --sys-prefix

# Set environment variables
ENV SHELL=/bin/bash

# Switch to the created user for any subsequent operations
USER $NB_UID

CMD ["python3"]
