from setuptools import setup, find_packages

setup(
    name="efficia_1",
    version="0.1.0",
    author="Antonín Tomeček",
    author_email="bambiliarda@gmail.com",
    description="EFFICIA-1: An efficient LLM architecture.",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/luxopes/efficia-1",
    project_urls={
        "Bug Tracker": "https://github.com/luxopes/efficia-1/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    python_requires=">=3.6",
    install_requires=[
        "torch",
        "einops",
    ],
)
