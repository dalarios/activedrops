import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="activedrops",
    version="0.0.1",
    author="David Larios",
    author_email="dalarios {at} caltech {dot} edu",
    description="This package contains all of the Python functionality for the activedrops project.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dalarios/activedrops",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)