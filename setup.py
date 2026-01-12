import setuptools


with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="polapy",
    version="0.0.3",
    author="cnavarreteliz",
    author_email="cnavarreteliz@gmail.com",
    description="PolaPy (Polarization for Python) is a collection of algorithmic implementations of polarization and electoral competition metrics in Python.",
    license_files=("LICENSE.md",),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/cnavarreteliz/polapy",
    packages=setuptools.find_packages(),
    python_requires=">=3.7",
    install_requires=[
        "numpy",
        "pandas",
        "scipy",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
