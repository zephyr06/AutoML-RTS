import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="automl_rts",
    version="v1.3.6",
    author="SE",
    author_email="swang666@vt.edu",
    description="Automl-RTS",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="3123",
    package_dir={"": "sources"},
    packages=setuptools.find_packages(where="sources"),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=['torch', 'numpy'],
    python_requires='>=3.6',
)
