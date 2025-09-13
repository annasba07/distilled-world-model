#!/usr/bin/env python3
"""
Setup script for Lightweight World Model
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="lightweight-world-model",
    version="0.1.0",
    author="Annas Bin Adil",
    author_email="",
    description="Lightweight Interactive World Model for Consumer GPUs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/annasba07/lightweight-world-model",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "einops>=0.7.0",
        "numpy>=1.24.0",
        "pillow>=10.0.0",
        "tqdm>=4.65.0",
        "matplotlib>=3.7.0",
        "tensorboard>=2.14.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.7.0",
            "isort>=5.12.0",
            "flake8>=6.1.0",
        ],
        "training": [
            "wandb>=0.15.0",
            "pytorch-lightning>=2.0.0",
            "accelerate>=0.24.0",
        ],
        "optimization": [
            "onnx>=1.14.0",
            "onnxruntime-gpu>=1.16.0",
        ],
        "api": [
            "fastapi>=0.100.0",
            "uvicorn>=0.23.0",
            "websockets>=11.0",
            "pydantic>=2.0.0",
            "fastapi-socketio>=0.0.10",
            "python-multipart>=0.0.6",
        ],
        "temporal": [
            "mamba-ssm>=1.2.0",
            "causal-conv1d>=1.1.0",
        ],
    },
    # Console scripts are disabled until stable CLIs are available
    # entry_points={
    #     "console_scripts": [
    #         "lwm-serve=src.api.server:main",
    #     ],
    # },
)
