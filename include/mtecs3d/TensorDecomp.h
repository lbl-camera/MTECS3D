#pragma once

#include "Common.h"

#include <array>
#include <string>
#include <vector>

namespace mtecs3d::detail
{
    /**
     * @brief The tensor decomposition step that find the matrix factor S and the rotational diffusion coefficient D_rot
     *
     */
    class TensorIsoRotProj
    {
    public:
        /**
         * @brief Construct a new TensorIsoRotProj object
         *
         * @param[in] delta_t_in Time interval between two image measurements
         * @param[in] dim_in Dimension of the input data
         */
        TensorIsoRotProj(const double delta_t_in, Dim *dim_in);

        /**
         * @brief Perform the tensor decomposition step
         * Implemented using the alternating least square method while the D_rot subproblem is a one-dimensional optimization problem
         * @param[in,out] G The input and output array of tensors, each of which is of size (num_basis[l], num_basis[l], num_delta_t).
         * @param[in,out] D_rot The initial guess and the output of the rotational diffusion coefficient
         */
        void Compute(TensorVec &G, double &D_rot);

        /**
         * @brief Set the parameters of the tensor decomposition
         *
         */
        void SetParameters()
        {
            max_iter = 5000;
            tol_rel = 1e-9;
            max_S_iter = 1000;
            tol_S_rel = 1e-9;
        }

    private:
        void WeightTensor();

        void UpdateS(const int ind_l);
        void UpdateR(const double D_rot);

        double UpdateDrot(double &D_rot);
        double ObjFuncDrot(const double D_rot);

        void ObtainTensor(TensorVec &G);

        TensorVec *G_ptr;

        std::vector<Eigen::MatrixXd> S;
        std::vector<Eigen::MatrixXd> R;
        std::vector<Eigen::ArrayXd> weights;

        int max_iter;
        double tol_rel;
        int max_S_iter;
        double tol_S_rel;

        std::vector<int> num_rank;
        Dim *dim;

        TensorVec fitted_G;
        double G_norm_obj;
        double delta_t = 1.0;
    };
} // namespace mtecs3d::detail