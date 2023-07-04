from pathlib import Path

from setuptools import find_packages, setup

with open(Path(__file__).resolve().parent / "README.md") as f:
    readme = f.read()

setup(
    name="multiscale_read",
    url="https://github.com/clbarnes/multiscale_read",
    author="Chris L. Barnes",
    description="Read multiscale arrays with various metadata standards",
    long_description=readme,
    long_description_content_type="text/markdown",
    package_dir={"": "src"},
    packages=find_packages(
        where="src",
        include=["multiscale_read*"]
    ),
    install_requires=[],
    python_requires=">=3.8, <4.0",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
    ],
    use_scm_version=True,
    setup_requires=["setuptools_scm"],
)
