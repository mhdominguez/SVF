from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Arguments marked as "Required" below must be included for upload to PyPI.
# Fields marked as "Optional" may be commented out.

setup(
    name='SVF',
    version='1.1',
    description='Statistical Vector Flow and the tissue propoagation scripts',
    long_description=long_description,
    url='https://github.com/mhdominguez/SVF',
    author='Leo Guignard, Martin Dominguez',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.10',
    ],
    install_requires=['scipy', 'numpy']
)
