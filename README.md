# SVF

This repository contains the Statistical Vector Flow (SVF) framework and the tissue propagation software proposed in the article *In toto imaging and reconstruction of post-implantation mouse development at the single-cell level*.  It has been updated for facile use running from the terminal, for use with complex path names, and there are minor bugs fixes.

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

`python SVF-prop.py csv-parameter-files/SVF-prop.csv`

`python tissue-bw-prop.py csv-parameter-files/tissue-bw-prop.csv`

The user should modify the parameter files prior to running with the correct information.  Note, the first line of each .csv parameter file contains the field delimeter used for the rest of the file; in other words, the user can use a different delimeter than `###` or `'` if desired.

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
