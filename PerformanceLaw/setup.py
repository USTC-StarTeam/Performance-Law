from setuptools import setup, find_packages

setup(
    name="PerformanceLaw",
    version="0.1.0",
    description="Estimate actual entropy of a sequence.",
    packages=find_packages(),
    install_requires=[],  # tqdm为可选项
    python_requires=">=3.6",
)