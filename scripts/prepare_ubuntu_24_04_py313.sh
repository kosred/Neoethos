#!/usr/bin/env bash
set -euo pipefail

# HPC MASTER PREP: Ubuntu 24.04 (Python 3.13 + CUDA 12.8)
# Automated Zero-Config Setup for the Forex AI Monster.

if [[ "${EUID:-$(id -u)}" -ne 0 ]]; then
  SUDO="sudo"
else
  SUDO=""
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "[*] Updating system and installing base build tools..."
${SUDO} apt-get update -y
${SUDO} apt-get install -y build-essential wget curl git \
    python3.13 python3.13-dev python3.13-venv \
    libsqlite3-dev pkg-config libatlas-base-dev gfortran \
    libgomp1 htop btop iotop sysstat

# 1. Rust-native indicator stack
echo "[*] Using Rust-native Vector-TA indicators; no external TA-Lib binary install required."

# 2. Python 3.13 Defaulting
echo "[*] Configuring Python 3.13 as the default interpreter..."
${SUDO} update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.13 1
${SUDO} update-alternatives --set python3 /usr/bin/python3.13

# 3. Pip Optimization
echo "[*] Upgrading pip and installing the unified pyproject runtime manifest..."
python3 -m pip install --upgrade pip setuptools wheel --user --break-system-packages

cd "${PROJECT_ROOT}"
python3 -m pip install -e ".[gpu]" --user --break-system-packages

# 4. Final Launch
echo "[*] Setup Complete. Launching Forex AI in 252-core HPC mode..."
exec python3 forex-ai.py --train
