from setuptools import setup, find_packages

setup(
    name='cuda-selector',
    version='0.1.0',
    description='A simple tool to select the optimal CUDA device based on memory, power, or utilization.',
    author='Samer Makni',
    author_email='samermakni@outlook.com',
    url='https://github.com/samermakni/cuda-selector',  # Replace with your repo URL
    packages=find_packages(),
    install_requires=[],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
