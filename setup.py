from setuptools import setup, find_packages

setup(
    name="OneModeltoMagnifyThemAll",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
    description="One Model to Magnify Them All: Efficient Scale-Invariant Histopathology via Conditional Normalization and Continuous Magnification Training",
    author="Agnieszka Florkowska",
    author_email="aflorkowska@agh.edu.pl",
)