import re

import setuptools
from setuptools import find_packages

with open("./autodistill_dinov2/__init__.py", "r") as f:
    content = f.read()
    # from https://www.py4u.net/discuss/139845
    version = re.search(r'__version__\s*=\s*[\'"]([^\'"]*)[\'"]', content).group(1)

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="autodistill-dinov2",
    version=version,
    author="Roboflow",
    author_email="support@roboflow.com",
    description="DINOv2 module for use with Autodistill",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/autodistill/autodistill-dinov2",
    install_requires=[
        "torch",
        "torchvision",
        "supervision",
        "numpy",
        "Pillow",
        "tqdm",
        "scikit-learn",
        "autodistill",
    ],
    packages=find_packages(exclude=("tests",)),
    extras_require={
        "dev": ["flake8", "black==22.3.0", "isort", "twine", "pytest", "wheel"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
