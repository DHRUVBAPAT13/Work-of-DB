FROM rockylinux:9

# 1. Enable EPEL and CRB (CodeReady Builder) for dev toolchains & NASM
RUN dnf install -y 'dnf-command(config-manager)' epel-release \
    && dnf config-manager --set-enabled crb \
    && dnf update -y

# 2. Install base essentials, compilers, and assemblers
RUN dnf install -y \
    git \
    curl \
    wget \
    tar \
    unzip \
    which \
    sudo \
    procps-ng \
    util-linux \
    shadow-utils \
    # Assembly & Linkers
    nasm \
    binutils \
    # C / C++ Toolchains & Debuggers
    gcc \
    gcc-c++ \
    make \
    cmake \
    gdb \
    # High-Level Runtimes
    golang \
    java-21-openjdk-devel \
    maven \
    dotnet-sdk-8.0 \
    python3 \
    python3-pip \
    python3-devel \
    nodejs \
    npm \
    # GUI editor & Terminal utilities
    gedit \
    tmux \
    htop \
    ripgrep \
    && dnf clean all \
    && rm -rf /var/cache/dnf

# 3. Install TypeScript and global JS toolchains
RUN npm install -g typescript ts-node yarn pnpm

# 4. Install Starship prompt for rich terminal colors & Git symbols
RUN curl -sS https://starship.rs/install.sh | sh -s -- -y

# 5. Create non-root 'vscode' user with passwordless sudo
RUN useradd -m -s /bin/bash vscode \
    && echo "vscode ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers.d/vscode \
    && chmod 0440 /etc/sudoers.d/vscode

USER vscode
WORKDIR /home/vscode

# Enable Starship and colorful terminal aliases
RUN echo 'eval "$(starship init bash)"' >> ~/.bashrc \
    && echo 'alias ls="ls --color=auto"' >> ~/.bashrc \
    && echo 'alias ll="ls -lah --color=auto"' >> ~/.bashrc