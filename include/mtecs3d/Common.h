#pragma once

#include <string>
#include <vector>

#if __has_include("mkl.h")
#include "mkl.h"
#elif __has_include("cblas.h") && __has_include("lapacke.h")
#include "cblas.h"
#include "lapacke.h"
#else
#error "Missing implementation of CBLAS and LAPACKE"
#endif

#include "utils.h"

#include <Eigen/Eigen>
#include <unsupported/Eigen/CXX11/Tensor>

namespace mtecs3d::detail
{
    using TensorVec = std::vector<Eigen::Tensor<double, 3>>;

    /**
     * @brief Struct of the dimension of the data
     *
     */
    struct Dim
    {
        int num_delta_t = 0; //!< Number of delta t
        int num_q = 0;       //!< Number of measured q
        int num_l = 0;       //!< Number of l in the Legendre expansion

        std::vector<double> snr_t{};  //!< Estimated snr of each time slice of the reduced correlation
        Eigen::VectorXd measured_q{}; //!< Measured q grid
        std::vector<int> num_basis{}; //!< Numbers of basis functions, number of the columns of S
    };
} // namespace mtecs3d
