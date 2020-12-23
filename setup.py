import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="numereval",
    version="0.0.1",
    author="Suraj Parmar",
    author_email="parmarsuraj99@gmail.com",
    description="A small package for evaluating numer.ai model locally",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/parmarsuraj99/numereval",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)