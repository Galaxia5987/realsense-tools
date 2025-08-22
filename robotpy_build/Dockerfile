# Use Ubuntu 24.04 as base
FROM ubuntu:24.04

# Set environment variables
ENV ROBOTPY_VERSION=2025.3.2.1
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Jerusalem

# Set timezone
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install system dependencies
RUN apt update && \
    apt install -y --no-install-recommends \
        curl \
        git \
        build-essential \
        zlib1g-dev \
        libncurses5-dev \
        libgdbm-dev \
        libnss3-dev \
        libssl-dev \
        libreadline-dev \
        libffi-dev \
        libbz2-dev \
        libsqlite3-dev \
        wget \
        llvm \
        liblzma-dev \
        tk-dev \
        libgdbm-compat-dev \
        ca-certificates && \
    apt clean && rm -rf /var/lib/apt/lists/*

# Build and install Python 3.10 from source
RUN cd /root && \
    curl -O https://www.python.org/ftp/python/3.10.12/Python-3.10.12.tgz && \
    tar -xvf Python-3.10.12.tgz && \
    cd Python-3.10.12 && \
    ./configure --enable-optimizations && \
    make -j$(nproc) && \
    make altinstall && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/local/bin/python3.10 2


# Set working directory
WORKDIR /root

# Clone robotpy repo
RUN git clone --branch ${ROBOTPY_VERSION} https://github.com/robotpy/mostrobotpy

# Install Python packages
WORKDIR /root/mostrobotpy
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install click numpy && \
    if [ -f rdev_requirements.txt ]; then python3 -m pip install -r rdev_requirements.txt; fi

# Default command
CMD ["bash"]
