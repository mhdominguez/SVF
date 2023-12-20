# SVF and svf2MaMuT

This repository contains the Statistical Vector Flow (SVF) package and the tissue propagation software published in *In toto imaging and reconstruction of post-implantation mouse development at the single-cell level*.  Now compatible with Python 3.x, it has been updated for easier terminal/console use, it can handle complex path names, is compatible with MaMuT v7, and there are other bugs fixes.  Note svf2MaMuT is now integrated into this SVF repository for easy setup.

## Description of the repository
Folders:
  - IO: The class `SpatialImage`, a container for images and input/output. When the right external libraries are installed (see bellow), can read tiff, hdf5, klb and inr images.
  - TGMMlibraries: The class `lineageTree`, a container for lineage trees and Statistical Vector Flow (SVF). Can read output data from TGMM.
  - csv-parameter-files: Example of parameterization csv files for each algorithms.
Python files to be run IN ORDER:
  - 1. SVF-prop.py: builds Statistical Vector Flow from a TGMM dataset.
  - 2. tissue-bw-prop.py: propagates tissue information from a manually annotated 3D image.
  - 3. SVF2MaMuT.py: exports results to MaMuT format for quantification and visualization.

## Basic usage
Each of the python scripts proposed here can be run from a terminal in the sequence:

`python3 /path/to/SVF/SVF-prop.py config-files/SVF-prop-config.txt`

`python3 /path/to/SVF/tissue-bw-prop.py config-files/tissue-bw-prop-config.txt`

`python3 /path/to/SVF/SVF2MaMuT.py config-files/svf2MM-config.txt`

The user should modify the parameter files prior to running with the correct information.

## Dependencies
Some dependecies are requiered:
  - general python dependecies:
    - numpy, scipy, pandas
  - SVF-prop.py:
     - TGMMlibraries installed (see TGMMlibraries README.md)
  - tissue-bw-prop.py:
    - TGMMlibraries installed (see TGMMlibraries README.md)
    - IO library installed (see IO README.md)
  - SVF2MaMuT.py:
    - TGMMlibraries installed (see TGMMlibraries README.md)
    - `Blank Dataset.tar.gz` contains an empty BigDataViewer dataset that is needed as a template.

## Quick install
Install IO and TGMMlibraries packages:
```shell
cd ~/Downloads
git clone https://github.com/GuignardLab/IO
git clone https://github.com/leoguignard/TGMMlibraries
cd IO
sudo python3 setup.py install
cd ../TGMMlibraries
sudo python3 setup.py install
cd ..
```

Install-free preparation of SVF:
```shell
git clone https://github.com/mhdominguez/SVF
```

