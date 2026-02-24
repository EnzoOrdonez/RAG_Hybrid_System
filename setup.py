"""Setup script for hybrid-rag-system."""

from setuptools import find_packages, setup

setup(
    name="hybrid-rag-system",
    version="0.1.0",
    author="Enzo Ordonez Flores",
    description=(
        "Hybrid RAG System for Cloud Documentation - "
        "Universidad de Lima thesis project"
    ),
    packages=find_packages(),
    python_requires=">=3.11",
    install_requires=[
        "pydantic>=2.0.0",
        "pyyaml>=6.0.0",
        "python-dotenv>=1.0.0",
        "tiktoken>=0.5.0",
        "requests>=2.31.0",
        "beautifulsoup4>=4.12.0",
        "lxml>=5.0.0",
        "rich>=13.0.0",
        "tqdm>=4.66.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
    ],
    extras_require={
        "dev": [
            "jupyterlab>=4.0.0",
            "matplotlib>=3.8.0",
            "seaborn>=0.13.0",
        ],
        "ml": [
            "sentence-transformers>=2.7.0",
            "datasketch>=1.6.0",
        ],
    },
)
