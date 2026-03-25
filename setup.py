from setuptools import setup, find_packages
import os

# Читаем README для длинного описания
long_description = ""
if os.path.exists("README.md"):
    with open("README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()

setup(
    name="muon-adamw8bit",
    version="0.5.0",
    author="recoilme",
    author_email="vadim-kulibaba@yandex.ru",
    description="Hybrid Muon + AdamW8bit optimizer for PyTorch. Handles matrices with Muon and scalars with 8-bit Adam.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/recoilme/muon_adamw8bit",
    project_urls={
        "Bug Tracker": "https://github.com/recoilme/muon_adamw8bit/issues",
    },
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "bitsandbytes>=0.41.0",
        "numpy"
    ],
)