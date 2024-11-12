#pragma once
#include "Common.h"

namespace mtecs3d::detail
{
    class BandLimitingProj
    {
    public:
        /**
         * @brief Construct a new Band Limiting Projector object
         * @param[in] diameter_in The estimated upper bound of the molecular diameter of the particles.
         * @param[in] dim_in The struct of the dimension of the data
         */
        BandLimitingProj(const double diameter_in, Dim *dim_in);

        /**
         * @brief Compute the downward step of the band-limiting projector
         * @param[in] B Tensor of size (num_delta_t, num_q, num_q, num_l, ), coefficients of the Legendre expansion
         * @param[out] G Vector of tensors of size (num_basis, num_basis, num_l, ), related to coefficients of the Fourier-Bessel expansion
         */
        void ComputeDownward(const Eigen::Tensor<double, 4> &B, TensorVec &G);

        /**
         * @brief Compute the upward step of the band-limiting projector
         * @param[in] G Vector of tensors of size (num_basis, num_basis, num_l, ), related to coefficients of the Fourier-Bessel expansion
         * @param[out] B Tensor of size (num_delta_t, num_q, num_q, num_l, ), coefficients of the Legendre expansion
         */
        void ComputeUpward(const TensorVec &G, Eigen::Tensor<double, 4> &B);

    private:
        Eigen::VectorXd dq;                      //!< Vector of size (num_q, ), integral weights of measured q points
        std::vector<Eigen::MatrixXd> band_V;     //!< Matrices of size (num_q, num_basis, ), the band-limiting basis
        std::vector<Eigen::MatrixXd> band_V_inv; //!< Matrices of size (num_basis, num_q, ), the matirces of the inverse of band_V
        Eigen::Tensor<double, 4> B_buffer;       //!< Tensor of size (num_q, num_q, num_delta_t, num_l)
        double diameter;                         //!< Estimated upper bound of the molecular diameter of the particles

        Dim *dim;
    };
} // namespace mtecs3d::detail