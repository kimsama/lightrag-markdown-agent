from setuptools import setup, find_packages

setup(
    name="lightrag-markdown-agent",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "streamlit>=1.24.0",
        "pydantic>=2.0.0",
        "python-dotenv>=1.0.0",
        "openai>=1.0.0",
        "pillow>=9.5.0",
        "markdown>=3.4.0",
        "beautifulsoup4>=4.12.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
    ],
    entry_points={
        "console_scripts": [
            "lightrag-md=lightrag_md_app:main",
        ],
    },
    author="Your Name",
    author_email="your.email@example.com",
    description="A LightRAG-based Markdown processing and querying system",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/lightrag-markdown-agent",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
) 