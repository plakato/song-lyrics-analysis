import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="song-lyrics-analysis-pa-tula", # Replace with your own username
    version="0.0.1",
    author="Patricia Brezinova",
    author_email="patricia@brezinovi.sk",
    description="Song lyrics analysis as master thesis.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/plakato/song-lyrics-analysis",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
