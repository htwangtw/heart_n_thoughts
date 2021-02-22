from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="heart_n_thought",
    version="0.1.0",
    description="ADIE ongoing thought analysis",
    license="MIT",
    long_description=long_description,
    long_description_content_type="text/markdown",  # Optional (see note above)
    maintainer="Hao-Ting Wang",
    maintainer_email="htwangtw@gmail.com",
    packages=find_packages(),
    install_requires=[
    ],  # external packages as dependencies
    python_requires=">=3.6",
)
