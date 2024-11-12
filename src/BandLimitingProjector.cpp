#include "mtecs3d/BandLimitingProjector.h"
#include "mtecs3d/Common.h"

#include <iostream>
#include <vector>

#include "gsl/gsl_sf_bessel.h"

using namespace Eigen;
using std::cout, std::endl;

namespace mtecs3d::detail
{
    /**
     * @brief Calculate the weight of each measured q point.
     *
     * @param[in] q Vector of size (num_q, ), measured q points.
     * @return Eigen::VectorXd, vector of size (num_q, ), integral weights of measured q points.
     */
    VectorXd calculate_dq(const VectorXd &q)
    {
        const int num_q = q.size();
        VectorXd dq(num_q);
        if (num_q == 1)
        {
            dq.setOnes();
            return dq;
        }

        dq(seq(1, last - 1)) = (q(seq(2, last)) - q(seq(0, last - 2))) / 2.0;
        double add_on = (q[num_q - 1] - q[0]) / (num_q - 1) / 2.0;
        dq[0] = (q[1] - q[0]) / 2.0 + add_on;
        dq[num_q - 1] = (q[num_q - 1] - q[num_q - 2]) / 2.0 + add_on;
        return dq;
    }

    /**
     * @brief Calculate the matrix of band-limiting basis functions.
     *
     * @param[in] l Frequency of the projector.
     * @param[in] num_basis Number of basis functions, number of the columns of S.
     * @param[in] q Vector of size (num_q, ), measured q.
     * @param[in] dq Vector of size (num_q, ), integral weights of measured q points.
     * @param[in] diameter Estimated upper bound of the molecular diameter of the particles.
     * @return Eigen::MatrixXd, matrix of size (num_q, num_basis, ), the matrix of the band-limiting basis functions.
     */
    MatrixXd band_limiting_matrices(const int l, const int num_basis, const VectorXd &q,
                                    const VectorXd &dq, const double diameter)
    {
        const int num_q = q.size();

        double *zeros = new double[num_basis];
        std::fill_n(zeros, num_basis, 0.0);
        for (int ind = 0; ind < num_basis; ind++)
        {
            zeros[ind] = gsl_sf_bessel_zero_Jnu(l + 0.5, ind + 1);
        }

        MatrixXd S(num_q, num_basis);
        for (int ind_q = 0; ind_q < num_q; ind_q++)
        {
            for (int ind_basis = 0; ind_basis < num_basis; ind_basis++)
            {
                double val = 4 * sqrt(2.0) * M_PI * pow(diameter, 1.5) * zeros[ind_basis] *
                             gsl_sf_bessel_jl(l, 2 * M_PI * q[ind_q] * diameter);
                S(ind_q, ind_basis) =
                    val * q[ind_q] * sqrt(dq[ind_q]) / (pow(zeros[ind_basis], 2) - pow(2 * M_PI * q[ind_q] * diameter, 2));
            }
        }

        delete[] zeros;
        return S;
    }

    BandLimitingProj::BandLimitingProj(const double diameter_in, Dim *dim_in)
    {
        dim = dim_in;
        diameter = diameter_in;
        const double q_max = dim->measured_q(last);
        dq = calculate_dq(dim->measured_q);
        dim->num_basis.resize(dim->num_l);
        band_V.resize(dim->num_l);
        band_V_inv.resize(dim->num_l);
        for (int ind_l = 0; ind_l < dim->num_l; ind_l++)
        {
            const int l_val = ind_l * 2 + 2;
            const int num_func = 2 * std::max(q_max * diameter - l_val / M_PI + 2.0, 1.0);
            MatrixXd basis_functions = band_limiting_matrices(l_val, num_func, dim->measured_q, dq, diameter);
            auto [U_band, s_band, VT_band] = mtecs3d::utils::linalg::SingValDecomp(basis_functions, mtecs3d::utils::linalg::SVDTYPE::ThinSVD);
            dim->num_basis[ind_l] = num_func;
            band_V[ind_l] = U_band.leftCols(dim->num_basis[ind_l]);
            band_V_inv[ind_l] = band_V[ind_l].transpose();
        }
        B_buffer.resize(dim->num_q, dim->num_q, dim->num_delta_t, dim->num_l);
    }

    void BandLimitingProj::ComputeDownward(const Tensor<double, 4> &B, TensorVec &G)
    {
#pragma omp parallel for collapse(2)
        for (int ind_l = 0; ind_l < dim->num_l; ind_l++)
        {
            for (int ind_t = 0; ind_t < dim->num_delta_t; ind_t++)
            {
                Map<MatrixXd> G_l_slice(&G[ind_l](0, 0, ind_t), dim->num_basis[ind_l], dim->num_basis[ind_l]);
                Map<MatrixXd> B_buffer_slice(&B_buffer(0, 0, ind_t, ind_l), dim->num_q, dim->num_q);
                for (int ind_q1 = 0; ind_q1 < dim->num_q; ind_q1++)
                {
                    for (int ind_q2 = 0; ind_q2 < dim->num_q; ind_q2++)
                    {
                        B_buffer_slice(ind_q1, ind_q2) = B(ind_l, ind_q1, ind_q2, ind_t);
                    }
                }
                B_buffer_slice.array().rowwise() *= dq.array().sqrt().transpose();
                B_buffer_slice.array().colwise() *= dq.array().sqrt();
                MatrixXd temp_G_l_slice = band_V_inv[ind_l] * B_buffer_slice * band_V_inv[ind_l].transpose();
                G_l_slice = (temp_G_l_slice + temp_G_l_slice.transpose()) / 2.0;
            }
        }
    }

    void BandLimitingProj::ComputeUpward(const TensorVec &G, Tensor<double, 4> &B)
    {
#pragma omp parallel for collapse(2)
        for (int ind_l = 0; ind_l < dim->num_l; ind_l++)
        {
            for (int ind_t = 0; ind_t < dim->num_delta_t; ind_t++)
            {
                Map<const MatrixXd> G_l_slice(&G[ind_l](0, 0, ind_t), dim->num_basis[ind_l], dim->num_basis[ind_l]);
                Map<MatrixXd> B_buffer_slice(&B_buffer(0, 0, ind_t, ind_l), dim->num_q, dim->num_q);
                B_buffer_slice = band_V[ind_l] * G_l_slice * band_V[ind_l].transpose();
                B_buffer_slice.array().rowwise() /= dq.array().sqrt().transpose();
                B_buffer_slice.array().colwise() /= dq.array().sqrt();
                for (int ind_q1 = 0; ind_q1 < dim->num_q; ind_q1++)
                {
                    for (int ind_q2 = 0; ind_q2 < dim->num_q; ind_q2++)
                    {
                        B(ind_l, ind_q1, ind_q2, ind_t) = B_buffer_slice(ind_q1, ind_q2);
                    }
                }
            }
        }
    }
} // namespace mtecs3d::detail
