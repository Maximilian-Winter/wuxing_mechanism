from setuptools import setup, find_packages

setup(
    name="wuxing_mechanism",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.7.0",
        "numpy>=1.19.0",
        "matplotlib>=3.3.0",
        "seaborn>=0.11.0",
        "pandas>=1.1.0",
    ],
    author="YourName",
    author_email="your.email@example.com",
    description="A framework for identifying and leveraging mechanism points in neural networks based on Wu Xing philosophy",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/wuxing_mechanism",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.7",
)