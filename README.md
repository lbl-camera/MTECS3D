# Multi-Tiered Estimation for Correlation Spectroscopy in 3D (MTECS3D)

[![DOI](https://zenodo.org/badge/887082775.svg)](https://doi.org/10.5281/zenodo.14090991)

_Multi-Tiered Estimation for Correlation Spectroscopy in 3D (MTECS3D)_ is an algorithm designed for estimating the rotational diffusion coefficient
from the angular-temporal cross-correlation of X-ray photon correlation spectroscopy images.
A relevant presentation can be [here](https://camera.lbl.gov/seminars) (titled _The Mathematics of Advanced Correlation Data Analysis: How to extract complex dynamics from next-generation correlation spectroscopy experiments_).

## Download

Clone via

```
git clone https://github.com/Tzi-Shi/MTECS3D.git
```

## Build

To build _MTECS3D_, you will need to have the following dependencies installed:

-   A **C++ compiler** supporting the C++17 standard or higher
-   **CMake** (minimum version 3.20)
-   **Eigen3**
-   **GNU Scientific Library (GSL)**
-   **HDF5 C Library**
-   **Intel MKL** (if not available, other implementations of CBLAS and LAPACKE are required)

Then the project can be built using the following command:

```shell script
# In the project directory
mkdir build  # It is recommended to create a separate build directory
cd build
cmake .. # If not in the build directory, replace ".." with the path to the CMakeLists.tx
make
```

After a successful build, an executable `bin/mtecs3d` will be created in the `bin` directory.

## Usage

All the arguments required by `bin/mtecs3d`, including parameters and the path to data files, are specified through the "../config.txt" file. An example `example/config.txt` is provided.
Then, _MTECS3D_ works in a two step manner. Here is an an example:

```shell command
# In the build directory
../bin/mtecs3d reduce ../example/config.txt
../bin/mtecs3d extract ../example/config.txt
```

The "reduce" option reduce the full angular-temporal cross-correlation into a compact form and the "extract" option estimate the rotational diffusion cofficient desired out of the reduced data.

The "config.txt" file should contain the following arguments for the "reduce" option:

-   **CorrFile**: The angular-temporal cross-correlation data.
-   **Lmax**: The truncation frequency of the Legendre expansion of the cross-correlation data.
-   **FlatEwaldSphere**: "1" means a flat Ewald sphere is assume and "0" if otherwise.
-   **Wavelength**: The wavelength of the X-ray in angstrom (Å).
-   **TruncationLimit**: Indices indicating the boundary of the masking for elimiating the peaks of the data at 0 and $\pi$. The part between the specified indices is the component that will be kept and used by the algorithm.
-   **ReducedCorrFile**: The reduced cross-correlation data file that will be input into the "extract" option.
-   **Verbose**: "1" indicates more verbose ouput and "0" otherwise.

For the "extract" option, the following arguments are required:

-   **ReducedCorrFile**: The reduced cross-correlation data file from which the rotational diffusion coefficient will be estimated.
-   **DeltaT**: The time difference between consecutive measurements of images in seconds.
-   **Diamter**: An estimate of the upper bound of the particle strucutre diameter in angstrom (Å). Should exceed the actual value. A rough estimate that is approximately tens of percent larger than the actual diamter is sufficient.
-   **MaxExtractIter**: Maximum iteration of the algorithm.
-   **Tol**: Stopping tolerance of the algorithm.
-   **Verbose**: "1" indicates more verbose ouput and "0" otherwise.

Lines in "config.txt" starting with "#" will be omitted.

The 'CorrFile' should be a HDF5 file containing two datasets, "/correlation" and "/q". For example:

```console
foo@bar:~$ h5ls -r example/cross-correlation.hdf5
/                        Group
/correlation             Dataset {16, 81, 81, 1024}
/q                       Dataset {81}
```

-   "Correlation" dataset: Contains the angular-temporal cross-correlation data calculated from the experimental images. Its dimension should be {num of lag times, number of measured q, number of measure q, number of angular coordinates}. Please note HDF5 stores the data in row-major order.
-   "q" dataset: Specifies the grid of the measured q in inverse of angstrom (Å⁻¹). Its dimension should be {number of measured q}.

<!-- For testing, an example data file `example/cross-correlation.hdf5` is provided. The ground truth of the rotational diffusion coefficient is $0.5$. An estimate with relative errors within a few percent should be given by our algorithm. -->
<!-- TODO upload the cross-correlation.hdf5 -->
<!-- Please refer to [our paper](TODO) for more details on some of the parameters above. -->

## Using as a library

In addition to the executable `bin/mtecs3d`, you can also use the functions provided by this project directly in your C++ code.
Currently, you need to manage the include path and the link line on your own. In future versions, we will provide the CMAKE config file to enable the MTECS3D package to be easily found by CMAKE.

### Including Header Files

In order to do so, you need to include the header file as, for example,

```c++
#include "include/mtecs3d.h"
```

### Linking Against the Library

The library can be obtained through the building process described above and linked as

```
lib/libmtecs3d_static.a # or through -lmtecs3d_static
```

In addition to this, you also need to include all the linking lines required by the above dependencies.

### Available Functions and Documentation

By including the header files and linking against the library as described above, you will have access to the following functions:

-   `mtecs3d::ReduceCorrelationData`
-   `mtecs3d::ExtractCoefficient`
-   `mtecs3d::detail::CorNoiseProj`
-   `mtecs3d::detail::BandLimitingProj`
-   `mtecs3d::detail::TensorIsoRotProj`

Documentation about the above functions can be found from where the functions are declared in the source codes.

<!-- Detailed documentation about the above functions and other components of this package can be found in [this documentation page](TODO). -->

<!-- TODO generate the documentation using Doxygen. -->

## About

---

Multi-Tiered Estimation for Correlation Spectroscopy in 3D (MTECS3D)
Copyright (c) 2024, The Regents of the University of California,
through Lawrence Berkeley National Laboratory (subject to receipt of
any required approvals from the U.S. Dept. of Energy). All rights reserved.

If you have questions about your rights to use or distribute this software,
please contact Berkeley Lab's Intellectual Property Office at
IPO@lbl.gov.

NOTICE. This Software was developed under funding from the U.S. Department
of Energy and the U.S. Government consequently retains certain rights. As
such, the U.S. Government has been granted for itself and others acting on
its behalf a paid-up, nonexclusive, irrevocable, worldwide license in the
Software to reproduce, distribute copies to the public, prepare derivative
works, and perform publicly and display publicly, and to permit others to do so.

---

\*\*\* License Agreement \*\*\*

GPL v3 License

Multi-Tiered Estimation for Correlation Spectroscopy in 3D (MTECS3D)
Copyright (c) 2024, The Regents of the University of California,
through Lawrence Berkeley National Laboratory (subject to receipt of
any required approvals from the U.S. Dept. of Energy). All rights reserved.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.

## Authors

_MTECS3D_ is developed by [Zixi Hu](https://crd.lbl.gov/divisions/amcr/mathematics-dept/math-for-experimental-data-analysis/people-of-math-for-experimental-data-analysis-group/zixi-hu/)
and
[Jeffrey Donatelli](https://crd.lbl.gov/divisions/amcr/mathematics-dept/math-for-experimental-data-analysis/people-of-math-for-experimental-data-analysis-group/jeff-donatelli/)
in [Center for Advanced Mathematics for Energy Research Applications (CAMERA)](https://camera.lbl.gov/)
and
[Math for Experimental Data Analysis (MEDA) Group](https://crd.lbl.gov/divisions/amcr/mathematics-dept/math-for-experimental-data-analysis/) at the [Lawrence Berkeley National Laboratory (LBNL)](https://www.lbl.gov/).

This work is supported by the Center of Advanced Mathematics for Energy Research Applications (CAMERA), funded by the US Department of Energy’s Office of Advanced Scientific Computing Research and Basic Energy Sciences under Contract DE-AC02-05CH11231. This research used resources of the National Energy Research Scientific Computing Center (NERSC), a Department of Energy Office of Science User Facility using NERSC award ASCR-ERCAP0027516.

## Bibtex

Please cite our papers using the following bibtex items:

```
@article{hu2021cross,
  title={Cross-correlation analysis of X-ray photon correlation spectroscopy to extract rotational diffusion coefficients},
  author={Hu, Zixi and Donatelli, Jeffrey J and Sethian, James A},
  journal={Proceedings of the National Academy of Sciences},
  volume={118},
  number={34},
  pages={e2105826118},
  year={2021},
  publisher={National Acad Sciences}
}
```
