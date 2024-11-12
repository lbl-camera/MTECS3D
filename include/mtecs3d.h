#pragma once

#include "mtecs3d/BandLimitingProjector.h"
#include "mtecs3d/Common.h"
#include "mtecs3d/CorrelationNoiseProjector.h"
#include "mtecs3d/TensorDecomp.h"

#include <array>

namespace mtecs3d
{
    /**
     * @brief Reduce the cross-correlation data into the form of U^TB
     * Use the singular value decomposition of the Legendre polynomial matrix to reudce the cross-correlation and
     * calculate the noise magnitude. Both of the input and output file should be in form of HDF5. The output file
     * contains the number of angles phi, the eigenvalues and the vt matrices of the SVD, the reduced cross-correlation and noise magnitude.
     * @param[in] input_filename Name of the input file containing the raw cross-correlation data
     * @param[in] output_filename Name of the output file storing the reduced cross-correlation data
     * @param[in] lmax Maximum value of L in the Legendre polynomial expansion
     * @param[in] flat_Ewald_sphere Flag indicating if the Ewald sphere is flat
     * @param[in] wavelength Wavelength of the beam
     * @param[in] truncation_limit Truncation indices of the angle phi to mask out the peak at phi=0 and phi=\pi
     * @param[in] verbose Verbose level
     */
    void ReduceCorrelationData(
        const char *input_filename,
        const char *output_filename,
        const int lmax,
        const bool flat_Ewald_sphere,
        const double wavelength,
        const std::array<int, 2> truncation_limit,
        const int verbose);

    /**
     * @brief Extract the rotational diffusion coefficient from the reduced cross-correlation data
     *
     * @param[in] reduced_correlation_filename Name of the file containing the reduced cross-correlation data
     * @param[in] delta_t Yime interval between the two image measurements
     * @param[in] diameter Estimated upper bound of the molecular diameter of the particles
     * @param[in] max_mtecs_iter Maximum number of iterations for the MTECS3D algorithm
     * @param[in] tol_mtecs Tolerance for the MTECS3D algorithm
     * @param[in] verbose Verbose level
     */
    void ExtractCoefficient(
        const char *reduced_correlation_filename,
        const double delta_t,
        const double diameter,
        const int max_mtecs_iter,
        const double tol_mtecs,
        const int verbose);
} // namespace mtecs3d