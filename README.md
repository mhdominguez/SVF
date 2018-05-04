# SVF

This repository contains the Statistical Vector Flow (SVF) framework and the tissue propagation software proposed in our Mouse-Atlas article.

## Description of the repository
Folders:
  - IO: The class `SpatialImage`, a container for images and input/output. When the right external libraries are installed (see bellow), can read tiff, hdf5, klb and inr images.
  - TGMMlibraries: The class `lineageTree`, a container for lineage trees and Statistical Vector Flow (SVF). Can read output data from TGMM.
  - csv-parameter-files: Example of parameterization csv files for each algorithms.
Python files:
  - SVF-prop.py: python script to build Statistical Vector Flow from a TGMM dataset.
  - tissue-bw-prop.py: python script to propagate tissue information from a manually annotated 3D image.

## Basic usage
Each of the python scripts proposed here can be run from a terminal in the following way:

`python SVF-prop.py`

`python tissue-bw-prop.py`

The user is then prompted to provide a parameter csv file (examples provided in the folder csv-parameter-files). The path to the parameter file should then be typed in the terminal.

## Dependencies
Some dependecies are requiered:
  - general python dependecies:
    - numpy, scipy, pandas
  - SVF-prop.py:
     - TGMMlibraries has to be installed (see TGMMlibraries README.md)
  - tissue-bw-prop.py:
    - TGMMlibraries has to be installed (see TGMMlibraries README.md)
    - IO library has to be installed (see IO README.md)

## Quick install
To quickly install the script so it can be call from the terminal and install too the common dependecies one can run
```shell
python setup.py install [--user]
```
Still will be remaining to install IO and TGMMlibraries packages.