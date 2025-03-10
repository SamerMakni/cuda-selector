from setuptools import setup, find_packages
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name='cuda-selector',
    version='0.1.5',
    description='A simple tool to select the optimal CUDA device based on memory, power, or utilization.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Samer Makni',
    author_email='samermakni@outlook.com',
    url='https://github.com/samermakni/cuda-selector', 
    packages=find_packages(),
    install_requires=[],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
