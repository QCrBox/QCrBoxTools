# QCrBoxTools

![Badge MPL 2-2](https://img.shields.io/badge/License-MPL_2.0-FF7139.svg?style=for-the-badge)

The QCrBox project requires the execution of different crystallographic tasks, such as the conversion / handling of cif files and the scripted execution of crystallographic software, which is collected in this python module.

## QCrBox Project Team
| Project Member   | Role  | Unit   |
|------------------|-------|--------|
| Dr. [Paul Niklas Ruth](https://github.com/Niolon)             |  Research Software Engineer (RSE) | [Advanced Research Computing, Durham University](https://www.durham.ac.uk/research/institutes-and-centres/advanced-research-computing/)                       |
| Dr. [Maximilian Albert](https://github.com/maxalbert)       |  Research Software Engineer (RSE) | [Research Software Group, University of Southampton ](https://rsgsoton.net/)                       |
| Prof. [Horst Puschmann](https://github.com/mulomulo) | Principal Investigator      | [OlexSys](https://www.olexsys.org/) |
| Prof. [Simon Coles](https://www.southampton.ac.uk/people/5wzkxv/professor-simon-coles)  | Principal Investigator      | [Department of Chemistry, University of Southampton](https://www.southampton.ac.uk/research/areas/chemistry) |

## Getting started

As QCrBoxTools requires cctbx the easiest way to install it is via conda.

```bash
conda install -f environment.yml
conda develop .
```

## Documentation
Most of the documentation is in code for the time being. The docstrings/type hints should be complete and informative if they are not please raise an issue. Look into the cif folder for functionality connected to cif manipulation and into the robots folder for building blocks to script Olex2 and Eval15.

The structure of the tests folder maps the qcrboxtools structure. If you want to see the functions in use you can also take a look there.

## Licence
The software is distributed under the MPL2.0 licence.