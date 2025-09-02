# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a PyTorch-based matrix factorization project that uses gradient descent to decompose matrix products. The main implementation is in `frob.py`.

## Development Environment

- Python version: 3.12
- Package manager: uv (version 0.6.17)
- Virtual environment: `.venv` directory

## Commands

### Running the main script
```bash
python frob.py
# or
./frob.py
```

### Managing dependencies
```bash
# Install dependencies
uv sync

# Add a new dependency
uv add <package_name>

# Update dependencies
uv lock --upgrade
```

### Activating virtual environment
```bash
source .venv/bin/activate
```

## Code Architecture

The project implements matrix factorization using PyTorch's automatic differentiation:

- **Matrix Factorization Model**: A neural network module (`MatrixFactorisation`) that learns to decompose a matrix product AB into its factors
- **Training Loop**: Uses gradient descent (SGD with momentum) to minimize MSE loss between predicted and target matrix products
- **Device Support**: Configured for CPU by default, with CUDA support available (currently commented out in frob.py:30-31)

Key parameters in the training process:
- Learning rate: 0.1
- Momentum: 0.9
- Epochs: 1000
- Loss function: MSELoss (with HuberLoss available as alternative)