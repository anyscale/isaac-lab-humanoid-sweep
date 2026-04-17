# IsaacLab base image + Anyscale runtime requirements.
ARG ISAACLAB_IMAGE=nvcr.io/nvidia/isaac-lab:2.3.2
ARG ANYSCALE_RAY_IMAGE=anyscale/ray:2.53.0-slim-py311-cu128
ARG RAY_VERSION=2.53.0

FROM ${ANYSCALE_RAY_IMAGE} AS anyscale-ray
FROM ${ISAACLAB_IMAGE}

SHELL ["/bin/bash", "-c"]
ARG RAY_VERSION

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV PYTHONUNBUFFERED=1
ENV TERM=xterm

USER root

# Anyscale-required system packages.
RUN apt-get update && apt-get install -y --no-install-recommends \
    sudo \
    supervisor \
    python3 \
    python3-venv \
    python3-pip \
    python-is-python3 \
    bash \
    openssh-server \
    openssh-client \
    rsync \
    zip \
    unzip \
    git \
    gdb \
    curl \
    && rm -rf /var/lib/apt/lists/*
RUN mkdir -p /var/run/sshd /etc/anyscale /opt/anyscale /tmp/anyscale /tmp/ray /mnt /anyscale/init

# Copy Anyscale runtime files from official Anyscale image.
COPY --from=anyscale-ray /opt/anyscale /opt/anyscale
COPY --from=anyscale-ray /home/ray/anaconda3 /home/ray/anaconda3

# Ensure a compatible `ray` user (uid=1000, gid=100) with passwordless sudo.
RUN set -eux; \
    if ! getent group 100 >/dev/null 2>&1; then groupadd -g 100 ray; fi; \
    if ! id -u ray >/dev/null 2>&1; then \
        uid_owner="$(getent passwd 1000 | cut -d: -f1 || true)"; \
        if [ -n "${uid_owner}" ] && [ "${uid_owner}" != "ray" ]; then \
            useradd -m -s /bin/bash -o -u 1000 -g 100 ray; \
        else \
            useradd -m -s /bin/bash -u 1000 -g 100 ray; \
        fi; \
    fi; \
    usermod -g 100 ray; \
    usermod -aG sudo ray; \
    printf '%s\n' \
      '# Anyscale startup uses sudo non-interactively.' \
      '# Match by uid, username, and sudo group for robustness.' \
      '#1000 ALL=(ALL:ALL) NOPASSWD:ALL' \
      'ray ALL=(ALL:ALL) NOPASSWD:ALL' \
      '%sudo ALL=(ALL:ALL) NOPASSWD:ALL' \
      > /etc/sudoers.d/90-anyscale-nopasswd; \
    chmod 0440 /etc/sudoers.d/90-anyscale-nopasswd; \
    mkdir -p /home/ray; \
    chown -R "$(id -u ray):$(id -g ray)" /home/ray /tmp/anyscale /tmp/ray
RUN su -s /bin/bash -c "sudo -n true" ray

# Install Anyscale runtime Python dependencies into Anyscale's own conda runtime.
# Keep Ray from the copied Anyscale image to avoid ABI mismatches.
RUN /home/ray/anaconda3/bin/python -m pip install --no-cache-dir \
    -r /opt/anyscale/runtime-requirements.txt \
    anyscale \
    packaging \
    boto3 \
    google \
    google-cloud-storage \
    jupyterlab \
    terminado \
    wandb

# Install wandb in Isaac Sim/IsaacLab Python (training-side logger).
RUN set -eux; \
    ISAACLAB_SH="$(command -v isaaclab.sh || true)"; \
    if [ -z "${ISAACLAB_SH}" ]; then \
      for p in /isaaclab/isaaclab.sh /workspace/isaaclab/isaaclab.sh; do \
        if [ -x "${p}" ]; then ISAACLAB_SH="${p}"; break; fi; \
      done; \
    fi; \
    if [ -z "${ISAACLAB_SH}" ]; then \
      echo "isaaclab.sh not found in base image"; exit 1; \
    fi; \
    CONDA_PREFIX= "${ISAACLAB_SH}" -p -m pip install --no-cache-dir wandb==0.24.1

# /isaac-sim may be a Docker VOLUME in the base image, meaning its contents
# vanish at runtime. Copy the Isaac Sim Python launcher + kit to a stable path.
RUN set -eux; \
    ISAACLAB_ROOT=""; \
    for p in /workspace/isaaclab /isaaclab; do \
      if [ -d "${p}/source" ]; then ISAACLAB_ROOT="${p}"; break; fi; \
    done; \
    ISAACSIM_REAL="$(readlink -f "${ISAACLAB_ROOT}/_isaac_sim")"; \
    echo "Isaac Sim real path: ${ISAACSIM_REAL}"; \
    cp -a "${ISAACSIM_REAL}" /opt/isaac-sim; \
    chmod -R a+rX /opt/isaac-sim; \
    mkdir -p /opt/isaac-sim/kit/data/Kit/Isaac-Sim/5.1/pip3-envs/default \
             /opt/isaac-sim/kit/cache/DerivedDataCache \
             /opt/isaac-sim/kit/cache/nv_shadercache \
             /opt/isaac-sim/kit/logs; \
    chmod -R a+rwX /opt/isaac-sim/kit/data /opt/isaac-sim/kit/cache /opt/isaac-sim/kit/logs; \
    echo "/opt/isaac-sim/python.sh" > /etc/isaacsim_python_path; \
    test -x /opt/isaac-sim/python.sh; \
    echo "Isaac Sim copied to /opt/isaac-sim with world-readable permissions"

# Workspace behavior expected by Anyscale.
RUN echo 'PROMPT_COMMAND="history -a"' >> /home/ray/.bashrc \
    && echo '[ -e ~/.workspacerc ] && source ~/.workspacerc' >> /home/ray/.bashrc
RUN touch /home/ray/.bashrc \
    && chown -R "$(id -u ray):$(id -g ray)" /home/ray \
    && chmod u+rw /home/ray/.bashrc

ENV HOME=/home/ray
ENV PATH=/home/ray/anaconda3/bin:${PATH}
ENV ANYSCALE_RAY_SITE_PKG_DIR=/home/ray/anaconda3/lib/python3.11/site-packages
USER ray
WORKDIR /home/ray
