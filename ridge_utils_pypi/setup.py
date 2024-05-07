from os import path

import setuptools

path_to_repo = path.abspath(path.dirname(__file__))

required_pypi = [
    'numpy',
    'scikit-learn',
    'pandas',
    'scipy',
]

setuptools.setup(
    # name="huth",
    name="ridge_utils",
    version="0.01",
    author="Huth lab modified by Chandan Singh",
    author_email="",
    description="",
    long_description='Installs ridge_utils used in the scaling laws paper by Huth lab: https://utexas.app.box.com/v/EncodingModelScalingLaws/folder/230420528915. This allows for loading the stored data files from there.',
    long_description_content_type="text/markdown",
    url="https://github.com/csinva/fmri",
    packages=setuptools.find_packages(
        exclude=['tests', 'tests.*', '*.test.*']
    ),
    python_requires='>=3.6',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=required_pypi,
)
