# Single-machine image for the KernelFoundry kernel generation algorithm on Intel GPUs.
# The algorithm builds, runs and profiles kernels directly inside this container.
#
# See devcontainer/devcontainer.json for using this image as a VS Code devcontainer.
#
# To use the image standalone you can use the ./kf wrapper which helps you composing the 
# docker run command with the right mounts and environment variables.
# To obtain the wrapper copy it from the repo checkout or from the image itself:
#   id=$(docker create kernelfoundry:local) && docker cp "$id:/kf" ./kf && docker rm "$id"
#
# The build context for this image is the repository root, not this directory, because the 
# image installs the surrounding checkout. Build from a checkout with:
#   docker build -f docker/Dockerfile -t kernelfoundry:local .
#
# Or without any checkout at all, letting Docker fetch the repository itself:
#   docker build -f docker/Dockerfile -t kernelfoundry:local \
#       https://github.com/isl-org/kernelfoundry.git#main
#
#   Behind a proxy, add:
#     --build-arg HTTP_PROXY=$http_proxy --build-arg HTTPS_PROXY=$https_proxy
#   The builder predeclares the proxy variables, so they need no ARG and their values stay
#   out of the image and out of the build cache key. Deliberately not turned into ENV, which
#   would bake one network's proxy into every container from this image; pass -e http_proxy
#   -e https_proxy -e no_proxy at run time instead, as the wrapper and the devcontainers do.
#   A docker client with a "proxies" block in ~/.docker/config.json covers both for you.
#
# Run the algorithm on the example task:
#
# Using the kf wrapper, the command is:
#   ./kf run task=tasks/example_custom task_origin=custom \
#     job_name=my_experiment gpu_arch=bmg language=SYCL
#
# The equivalent docker run command is:
#   mkdir -p runs
#   docker run --rm -it \
#     --user "$(id -u):$(id -g)" -e USER="$USER" \
#     --device /dev/dri --group-add "$(getent group render | cut -d: -f3)" \
#     -e OPENAI_API_KEY \
#     -v "$PWD/runs:/opt/kernelfoundry/runs" \
#     kernelfoundry:local \
#     python -m kernelfoundry.algorithm run task=tasks/example_custom task_origin=custom \
#       job_name=my_experiment gpu_arch=bmg language=SYCL
#
# --user keeps the generated files owned by you instead of root; drop it if you do not
# need that. --group-add render grants access to the GPU render node. USER is the name
# jobs are tagged with in the database and the UI.
#
# To work on your own task, mount it and point `task` at the mount:
#     -v "$PWD/my_task:/tasks/my_task"   ...   task=/tasks/my_task
#
# Instead of passing -e for every API key, you can mount an env file that the algorithm
# picks up on startup:
#     -v "$PWD/.env:/opt/kernelfoundry/.env:ro"
#
# OPENAI_BASE_URL and ANTHROPIC_BASE_URL select an endpoint other than the vendor one, a
# company LLM gateway for instance. Such a gateway is usually signed by a root certificate
# this image does not carry, so share the host's trust store as well:
#     -e OPENAI_BASE_URL -e no_proxy \
#     -v /etc/ssl/certs/ca-certificates.crt:/etc/ssl/certs/ca-certificates.crt:ro
# The wrapper and the devcontainers do both of these for you.
#
# Inspect the results in the UI on http://localhost:8885:
#   docker run --rm -it -p 8885:8885 -v "$PWD/runs:/opt/kernelfoundry/runs" \
#     kernelfoundry:local python start_gui.py
#
# unitrace and VTune read GPU performance counters, which the host kernel only hands out
# to non-root callers after:
#   sudo sh -c 'echo 0 > /proc/sys/dev/xe/observation_paranoid'     # xe driver
#   sudo sh -c 'echo 0 > /proc/sys/dev/i915/perf_stream_paranoid'   # i915 driver
#   sudo sh -c 'echo 0 > /proc/sys/kernel/kptr_restrict'            # VTune
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

# Install python3.12
RUN apt-get update && apt-get install -y \
    python3.12 \
    python3-pip \
    python3-venv \
    python3-dev \
    build-essential \
    ca-certificates \
    software-properties-common \
    wget \
    curl \
    git \
    unzip \
    pkg-config \
    cmake \
    binutils \
    patchelf \
    libclang-20-dev \
    graphviz \
    libgraphviz-dev \
    && rm -rf /var/lib/apt/lists/*

# Ensure python3 points to 3.12
RUN ln -sf /usr/bin/python3.12 /usr/bin/python3

###### Intel GPU runtime

# Add the intel-graphics PPA.
RUN add-apt-repository -y ppa:kobuk-team/intel-graphics
# Install the compute-related packages.
RUN apt-get update && apt-get install -y \
    libze-intel-gpu1 \
    libze1 \
    libze-dev \
    intel-metrics-discovery \
    intel-metrics-library \
    intel-opencl-icd \
    intel-ocloc \
    intel-gsc \
    clinfo \
    opencl-headers \
    ocl-icd-opencl-dev \
    && rm -rf /var/lib/apt/lists/*

###### oneAPI

# Download the key to the system keyring.
RUN wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
    | gpg --dearmor > /usr/share/keyrings/oneapi-archive-keyring.gpg

# Add the signed entry to the apt sources.
RUN echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" \
    > /etc/apt/sources.list.d/oneAPI.list

# Install the toolkit (icpx, xptifw) with explicit vtune version pinning.
RUN apt-get update && apt-get install -y \
    intel-oneapi-base-toolkit-2025.2 \
    intel-oneapi-vtune=2026.1.0-13 \
    && rm -rf /var/lib/apt/lists/*

###### unitrace (default profiler for SYCL kernels)

ARG PTI_GPU_REF=master
RUN . /opt/intel/oneapi/setvars.sh --force && \
    git clone -b ${PTI_GPU_REF} --depth 1 https://github.com/intel/pti-gpu.git /opt/pti-gpu && \
    cmake -S /opt/pti-gpu/tools/unitrace -B /opt/pti-gpu/tools/unitrace/build \
        -DCMAKE_BUILD_TYPE=Release -DBUILD_WITH_MPI=OFF && \
    cmake --build /opt/pti-gpu/tools/unitrace/build --parallel

###### Python environment

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:/opt/pti-gpu/tools/unitrace/build:$PATH"

RUN python -m pip install --no-cache-dir --upgrade pip

# PyTorch built against the Intel XPU backend
ARG TORCH_VERSION=2.9.1
RUN python -m pip install --no-cache-dir torch==${TORCH_VERSION} \
    --extra-index-url https://download.pytorch.org/whl/xpu

###### KernelFoundry

ENV KERNELFOUNDRY_HOME=/opt/kernelfoundry

# The whole checkout is copied, not just the package, because the algorithm resolves its
# hydra configs and task definitions relative to the .project-root marker above the
# package. See .dockerignore for what is left out.
COPY . ${KERNELFOUNDRY_HOME}

WORKDIR ${KERNELFOUNDRY_HOME}

# The pip cache is a build cache mount, so it speeds up rebuilds after a source edit
# without ending up in the image.
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install -e ".[algo,ui,mcp]" \
    matplotlib

# Reinstall pyopencl from source so it links against the system OpenCL loader.
RUN python -m pip uninstall -y pyopencl && \
    PIP_NO_BINARY=pyopencl python -m pip install --no-cache-dir pyopencl

# Default location of the SQLite kernel database and the job logs. Mount a host
# directory here to keep generated kernels across container runs. World-writable so the
# container can also be started with --user <uid>:<gid>.
RUN mkdir -p ${KERNELFOUNDRY_HOME}/runs ${KERNELFOUNDRY_HOME}/results && \
    chmod 0777 ${KERNELFOUNDRY_HOME}/runs ${KERNELFOUNDRY_HOME}/results

# Keep the build caches out of $HOME, which an arbitrary --user uid cannot write to.
ENV TORCH_EXTENSIONS_DIR=/tmp/torch_extensions
ENV XDG_CACHE_HOME=/tmp/.cache

# Jobs are tagged with getpass.getuser(), which fails for a --user uid that has no
# /etc/passwd entry. Override with -e USER=<name> to attribute jobs to yourself.
ENV USER=kernelfoundry

###### Interactive and devcontainer use

# Ubuntu 24.04 ships an unprivileged "ubuntu" user. Devcontainers run as that user so that
# files created in the mounted workspace belong to the host account rather than to root.
# It still needs sudo occasionally, because the venv and the checkout below /opt are
# root-owned: installing an extra package is `sudo /opt/venv/bin/pip install <pkg>`.
# Installed here rather than with the other packages so that editing this does not
# invalidate the oneAPI and unitrace layers above.
#
# The marker file silences the "To run a command as administrator" banner that Ubuntu's
# /etc/bash.bashrc prints in every new shell once sudo is installed. It also avoids the
# `groups` call in that banner, which warns about the render group id having no name here.
RUN apt-get update && apt-get install -y sudo && rm -rf /var/lib/apt/lists/* && \
    echo 'ubuntu ALL=(ALL) NOPASSWD:ALL' > /etc/sudoers.d/ubuntu && \
    chmod 0440 /etc/sudoers.d/ubuntu && \
    install -o ubuntu -g ubuntu /dev/null /home/ubuntu/.sudo_as_admin_successful

# Install copilot cli which is a dependency for the agentic mode
RUN curl -fsSL https://gh.io/copilot-install | bash


# Devcontainer terminals start with docker exec, which bypasses ENTRYPOINT, so the oneAPI
# and virtualenv setup has to be repeated in the shell startup files. profile.d covers
# login shells and .bashrc covers interactive non-login shells, which skip profile.d.
#
# devcontainer.json passes the __KF_UNSET__ sentinel for host variables that are not
# defined, since an empty value would still shadow .env file entries. The loop that strips
# it runs before the proxy fallbacks so that no sentinel is copied.
RUN printf '%s\n' \
        'source /opt/intel/oneapi/setvars.sh --force >/dev/null' \
        'export PATH="/opt/venv/bin:$PATH"' \
        'for _kf_var in $(compgen -e); do' \
        '    if [[ "${!_kf_var}" == "__KF_UNSET__" ]]; then unset "$_kf_var"; fi' \
        'done' \
        'unset _kf_var' \
        '[[ -n "${HTTP_PROXY:-}" && -z "${http_proxy:-}" ]] && export http_proxy="$HTTP_PROXY"' \
        '[[ -n "${HTTPS_PROXY:-}" && -z "${https_proxy:-}" ]] && export https_proxy="$HTTPS_PROXY"' \
        '[[ -n "${NO_PROXY:-}" && -z "${no_proxy:-}" ]] && export no_proxy="$NO_PROXY"' \
        > /etc/profile.d/10-kernelfoundry.sh && \
    for rc in /root/.bashrc /home/ubuntu/.bashrc; do \
        echo '. /etc/profile.d/10-kernelfoundry.sh' >> "$rc"; \
    done

# The kf wrapper drives docker from the host, we place it at /kf to make it short to lift
#  out of the image:
#   id=$(docker create kernelfoundry:local) && docker cp "$id:/kf" ./kf && docker rm "$id"
COPY docker/kf /kf

# We do the same with the devcontainer config for this image:
#   id=$(docker create kernelfoundry:local)
#   docker cp "$id:/.devcontainer" . && docker rm "$id"
COPY docker/devcontainer/devcontainer.json /.devcontainer/devcontainer.json

# A copy taken out of the image has to point back at the image it came from, otherwise it
# resolves to whatever the checked-in template names. Whoever publishes should pass the tag
# they push, for example
#   --build-arg KF_IMAGE_REF=ghcr.io/isl-org/kernelfoundry:v0.3.0
ARG KF_IMAGE_REF=ghcr.io/isl-org/kernelfoundry:latest
RUN sed -i "s|^\( *\)\"image\": \".*\"|\1\"image\": \"${KF_IMAGE_REF}\"|" \
        /.devcontainer/devcontainer.json && \
    grep -q "\"image\": \"${KF_IMAGE_REF}\"" /.devcontainer/devcontainer.json

COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["bash"]
