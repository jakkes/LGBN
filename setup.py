import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()
with open("VERSION", "r") as f:
    version = f.read()

setuptools.setup(
    name="lgbn",
    version=version,
    author="Jakob Stigenberg",
    description="Linear Gaussian Belief Networks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jakkes/LGBN",
    packages=setuptools.find_packages(),
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
    install_requires=[],
    python_requires=">=3.7",
)

# Publish
# python setup.py sdist bdist_wheel
# twine upload dist/*
