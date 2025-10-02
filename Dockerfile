# 1. Start from the official OpenShift AI internal base image
# Dockerfile

FROM openshift-image-registry.apps.cluster-dl2xj.dl2xj.sandbox472.opentlc.com/redhat-ods-applications/runtime-datascience

# 2. Switch to the root user to install packages
# Switch to the root user to install packages
USER root

# 3. Copy requirements.txt and install third-party packages
# This step is done first to leverage Docker layer caching.
# This layer will only be rebuilt if requirements.txt changes.
# === Part 1: Install Python packages from requirements.txt ===
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt

# 4. Copy and install your custom 'utils' package
# This layer will be rebuilt whenever your custom code in utils/ changes.
# === Part 2: Copy and install our custom Python package ===
COPY ./cez_custom_package /tmp/cez_custom_package
COPY ./utils /tmp/utils
This makes `from utils import ...` work everywhere.
RUN pip install /tmp/cez_custom_package && \
    rm -rf /tmp/cez_custom_package
RUN pip install /tmp/utils && \
    rm -rf /tmp/utils

# 5. Switch back to the default non-privileged user for security
# Switch back to the default notebook user for security
USER 1001
