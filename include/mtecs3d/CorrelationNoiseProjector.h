#pragma once
#include "Common.h"
#include "optimize.h"

#include <tuple>

namespace mtecs3d::detail
{
    /**
     * @brief The correlation noise projector that updates the B*q*q'
     * This projector utilizes the Tikhonov regularization to seek the minimum perturbation to Bqq such that the updated coefficients
     * makes the cross-correlation within the estimated uncertainty of the raw input data
     */
    class CorNoiseProj
    {
    public:
        /**
         * @brief Construct a new Correlation Noise Projector object
         * Initialize the correlation noise projector
         * The data file includes at least six datasets, each of which contains the measured q, numbers of angles phi,
         * the vt matrices, the eigenvalues, the reduced cross-correlation and the squred noise mangnitude.
         * This function normalize the reduced correlation data and the eigenvalues so that they can be feed into the
         * standard Tikhonov regularization, and calculate the parameter of Tikhonov regularization as well.
         * @param[in] input_filename Name of the input file
         * @param[in] dim_in Dimension of the input data
         * @param[in] verbose Verbose level
         */
        CorNoiseProj(const char *input_filename, Dim *dim_in, const int verbose);

        /**
         * @brief Reset lambda, tol, max_iter of the Tikhonov regularization solver
         */
        void SetParameters()
        {
            solver.lambda_ = 1e20;
            solver.tol_ = 1e-15;
            solver.max_iter_ = 1000;
        }

        /**
         * @brief The correlation noise projector that updates the B*q*q'
         * This projector utilizes the Tikhonov regularization to seek the minimum perturbation to Bqq such that the updated coefficients
         * makes the cross-correlation within the estimated uncertainty of the raw input data
         * @param[in,out] B Tensor of size (num_l, num_q, num_q, num_delta_t, ), coefficients to be updated
         */
        void Compute(Eigen::Tensor<double, 4> &B);

    private:
        Eigen::Tensor<double, 4> eigvals;      //!< Tensor of size (num_l, num_q, num_q, num_delta_t, ), obtained eigenvalues
        Eigen::Tensor<double, 4> reduced_corr; //!< Tensor of size (num_l, num_q, num_q, num_delta_t, ), obtained reduced cross-correlation data
        Eigen::Tensor<double, 5> vt;           //!< Tensor of size (num_l, num_l, num_q, num_q, num_delta_t, ), obtained vt matrices

        double tau;                                          //!< The parameter of Tikhonov regularization
        mtecs3d::utils::optimize::VectorizedTikhonov solver; //!< The Tikhonov regularization solver

        Eigen::Tensor<double, 4> C;       //!< Tensor of size (num_l, num_q, num_q, num_delta_t, ). Constant vector in the regularization
        Eigen::Tensor<double, 4> delta_B; //!< Tensor of size (num_l, num_q, num_q, num_delta_t, ). Update to the tensor B

        Dim *dim;
    };
} // namespace mtecs3d::detail