"""Setup script for chat-from-scratch package."""

from setuptools import setup, find_packages

setup(
    name="chat-from-scratch",
    version="0.1.0",
    description="Train a dialogue model from scratch — for learning, not SOTA",
    author="",
    packages=find_packages(where="."),
    package_dir={"": "."},
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.4.0",
        "tokenizers>=0.19.0",
        "datasets>=2.20.0",
        "pyyaml>=6.0",
        "tqdm>=4.66.0",
    ],
)
