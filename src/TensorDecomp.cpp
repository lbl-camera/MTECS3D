#include "mtecs3d/TensorDecomp.h"
#include "mtecs3d/Common.h"

#include "optimize.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>

using namespace Eigen;
using std::cout, std::endl;

namespace mtecs3d::detail
{
    TensorIsoRotProj::TensorIsoRotProj(const double delta_t_in, Dim *dim_in)
        : delta_t(delta_t_in)
    {
        dim = dim_in;
        num_rank.resize(dim->num_l);
        for (int ind_l = 0; ind_l < dim->num_l; ind_l++)
        {
            num_rank[ind_l] = std::min(4 * ind_l + 5, dim->num_basis[ind_l]);
        }

        G_ptr = nullptr;

        S.resize(dim->num_l);
        R.resize(dim->num_l);
        weights.resize(dim->num_l);
        for (int ind_l = 0; ind_l < dim->num_l; ind_l++)
        {
            S[ind_l] = MatrixXd::Zero(dim->num_basis[ind_l], num_rank[ind_l]);
            R[ind_l] = MatrixXd::Zero(dim->num_delta_t, 1);
            R[ind_l].row(0).setConstant(1.0);
            weights[ind_l] = ArrayXd::Ones(dim->num_delta_t);
            for (int ind_t = 0; ind_t < dim->num_delta_t; ind_t++)
            {
                weights[ind_l](ind_t) = dim->snr_t[ind_t];
            }
        }

        SetParameters();
    }

    void TensorIsoRotProj::Compute(TensorVec &G, double &D_rot)
    {
        G_ptr = &G;
        WeightTensor();
        G_norm_obj = 0.0;
        for (int ind_l = 0; ind_l < dim->num_l; ind_l++)
        {
            double val = mtecs3d::utils::linalg::Norm(G[ind_l]);
            G_norm_obj += val * val;
        }

        if (D_rot <= 0.0)
        {
            for (int ind_l = 0; ind_l < dim->num_l; ind_l++)
            {
                R[ind_l].setZero();
                R[ind_l].row(0).setConstant(1.0);
            }
        }

        int iter = 0;
        double val = 1.0;
        double change_val = 1.0;
        do
        {
            iter++;
            double prev_val = val;

            for (int ind_l = 0; ind_l < dim->num_l; ind_l++)
            {
                UpdateS(ind_l);
            }

            val = UpdateDrot(D_rot);
            UpdateR(D_rot);

            change_val = std::abs(val - prev_val);
        } while (iter < max_iter && change_val > tol_rel);

        for (int ind_l = 0; ind_l < dim->num_l; ind_l++)
        {
            R[ind_l].array().colwise() /= weights[ind_l];
            ObtainTensor(G);
        }

        G_ptr = nullptr;
    }

    void TensorIsoRotProj::WeightTensor()
    {
        for (int ind_l = 0; ind_l < dim->num_l; ind_l++)
        {
            if (dim->num_basis[ind_l] >= num_rank[ind_l])
            {
                for (int ind_t = 0; ind_t < dim->num_delta_t; ind_t++)
                {
                    Map<MatrixXd> mat((*G_ptr)[ind_l].data() + ind_t * dim->num_basis[ind_l] * dim->num_basis[ind_l],
                                      dim->num_basis[ind_l], dim->num_basis[ind_l]);
                    mat *= weights[ind_l](ind_t);
                }
                R[ind_l].array().colwise() *= weights[ind_l];
            }
        }
    }

    void TensorIsoRotProj::UpdateS(const int ind_l)
    {
        const int I = dim->num_basis[ind_l];
        const int K = dim->num_delta_t;
        const int rank = num_rank[ind_l];
        MatrixXd mat = MatrixXd::Zero(I, I);
        for (int ind_k = 0; ind_k < K; ind_k++)
        {
            cblas_daxpy(mat.size(), R[ind_l](ind_k, 0), (*G_ptr)[ind_l].data() + ind_k * mat.size(), 1, mat.data(), 1);
        }

        lapack_int num_eigvals;
        VectorXd eigvals = VectorXd::Zero(I);
        std::vector<lapack_int> ifail(I, 0);
        LAPACKE_dsyevx(LAPACK_COL_MAJOR, 'V', 'I', 'U', I, mat.data(), I, 0.0, 0.0, I - rank + 1, I,
                       2.0 * LAPACKE_dlamch('S'), &num_eigvals, eigvals.data(), S[ind_l].data(), I, ifail.data());
        for (int ind_r = 0; ind_r < rank; ind_r++)
        {
            double col_scale = std::sqrt(std::max(eigvals[ind_r], 0.0));
            S[ind_l].col(ind_r) *= col_scale;
        }
        S[ind_l] /= cblas_dnrm2(K, R[ind_l].data(), 1);
        Map<MatrixXd> mat_map((*G_ptr)[ind_l].data(), I, I);
    }

    void TensorIsoRotProj::UpdateR(const double D_rot)
    {
        for (int ind_l = 0; ind_l < dim->num_l; ind_l++)
        {
            for (int ind_k = 0; ind_k < dim->num_delta_t; ind_k++)
            {
                const int l_val = 2 * ind_l + 2;
                R[ind_l](ind_k, 0) = std::exp(-l_val * (l_val + 1) * ind_k * delta_t * D_rot);
            }
            R[ind_l].array().colwise() *= weights[ind_l];
        }
    }

    double TensorIsoRotProj::UpdateDrot(double &D_rot)
    {
        using namespace mtecs3d::utils::optimize;

        D_rot = std::max(D_rot, 0.0);
        // double current_fval = ObjFuncDrot(D_rot);
        double ret_val = 0.0;
        // double left_bound = 0.0;
        // double right_bound = D_rot + 1.0;
        // const int search_bound_iter = 100;
        // bool left_bound_found = false;
        // for (int ind_iter = 0; ind_iter < search_bound_iter; ind_iter++)
        // {
            // double val = ObjFuncDrot(left_bound);
            // if (val > current_fval)
            // {
                // left_bound_found = true;
                // break;
            // }
            // if (left_bound >= 0.0)
            // {
                // left_bound -= 1.0;
            // }
            // else
            // {
                // left_bound *= 2.0;
            // }
        // }
        // bool right_bound_found = false;
        // for (int ind_iter = 0; ind_iter < search_bound_iter; ind_iter++)
        // {
            // double val = ObjFuncDrot(right_bound);
            // if (val > current_fval)
            // {
                // right_bound_found = true;
                // break;
            // }
            // else
            // {
                // D_rot = right_bound;
                // current_fval = val;
            // }
            // right_bound *= 2.0;
        // }

        // if (left_bound_found && right_bound_found && D_rot < right_bound && D_rot > left_bound)
        // {
            // OneDimMin(D_rot, std::bind(&TensorIsoRotProj::ObjFuncDrot, this, std::placeholders::_1),
                      // ONEMINIMIZER::brent, left_bound, right_bound, 1e-12, 0.0, 10000, ret_val);
        // }
        // else
        // {
            auto obj_func = [&](const double *D_rot)
            {
                return ObjFuncDrot(*D_rot);
            };
            std::vector<double> nmsimplex_stepsize{0.2};
            NonGradMultiMin(1, &D_rot, obj_func, FMINIMIZER::nmsimplex2, nmsimplex_stepsize.data(), 1e-12, 100000,
                            ret_val);
        
        return ret_val;
    }

    double TensorIsoRotProj::ObjFuncDrot(const double D_rot)
    {
        if (D_rot < 0.0)
        {
            return std::numeric_limits<double>::infinity();
        }

        UpdateR(D_rot);
        ObtainTensor(fitted_G);

        double ret_val = 0.0;
        for (int ind_l = 0; ind_l < dim->num_l; ind_l++)
        {
            double val = mtecs3d::utils::linalg::Dist(fitted_G[ind_l], (*G_ptr)[ind_l]);
            ret_val += val * val;
        }
        ret_val /= G_norm_obj;

        return ret_val;
    }

    void TensorIsoRotProj::ObtainTensor(TensorVec &G)
    {
        G.resize(dim->num_l);
        for (int ind_l = 0; ind_l < dim->num_l; ind_l++)
        {
            G[ind_l].resize(dim->num_basis[ind_l], dim->num_basis[ind_l], dim->num_delta_t);
            G[ind_l].setConstant(0.0);
            Map<MatrixXd> mat(G[ind_l].data(), dim->num_basis[ind_l], dim->num_basis[ind_l]);
            mat.noalias() = S[ind_l] * S[ind_l].transpose();
            for (int ind_k = 1; ind_k < dim->num_delta_t; ind_k++)
            {
                cblas_daxpy(mat.size(), R[ind_l](ind_k, 0), mat.data(), 1, G[ind_l].data() + ind_k * mat.size(), 1);
            }
            mat *= R[ind_l](0, 0);
        }
    }
} // namespace mtecs3d::detail
