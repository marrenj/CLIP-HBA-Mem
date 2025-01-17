from setuptools import setup, find_packages

setup(
    name="cliphba",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.8, <3.9",
    install_requires=[
        "torch==1.10.2",
        "torchvision",
        "openmim",
    ],
    extras_require={
        "dev": [
            "mmcv-full==1.5.0",
            # Add all dependencies from requirements.txt here if they are not already in install_requires
        ]
    },
    entry_points={
        "console_scripts": [
            # Add any console script entry points here if your package provides any executables
        ]
    },
    # Metadata
    author="stephenczhao",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',  # This requires setuptools>=38.6.0
    url="https://github.com/stephenczhao/CLIP-HBA-Finetune",
    # Add any necessary classifiers here such as development status, environment, framework
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)

