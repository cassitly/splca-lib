# File structure for pip-installable library:
# splca/
# ├── setup.py
# ├── README.md
# ├── requirements.txt
# ├── splca/
# │   ├── __init__.py
# │   ├── core.py
# │   ├── layers.py
# │   ├── predictors.py
# │   ├── modulation.py
# │   ├── models/
# │   │   ├── __init__.py
# │   │   ├── text.py
# │   │   ├── vision.py
# │   │   └── audio.py
# │   └── utils.py
# ├── examples/
# │   ├── mnist_demo.py
# │   ├── text_classification_demo.py
# │   └── audio_classification_demo.py
# └── tests/
#     └── test_core.py

# ============================================================================
# setup.py
# ============================================================================
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="splca",
    version="0.1.0",
    author="Cassitly",
    author_email="cassitly@nakashireyumi.com",
    description="Self-Predictive Local Credit Assignment: Biologically-plausible learning for neural networks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/cassitly/splca",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "torchaudio>=2.0.0",
        "numpy>=1.21.0",
        "tqdm>=4.62.0",
        "matplotlib>=3.4.0",
    ],
    extras_require={
        "dev": ["pytest>=7.0.0", "black>=22.0.0", "flake8>=4.0.0"],
    },
)