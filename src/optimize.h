#pragma once

#include <functional>
#include <vector>

namespace mtecs3d::utils::optimize
{
    using OBJ_FUNC = std::function<double(const double *)>;   //!< val = func(x)
    using OBJ_ONE_FUNC = std::function<double(const double)>; //!< val = one_func(x)

    enum class FMINIMIZER
    {
        nmsimplex2,
        nmsimplex,
        nmsimplex2rand
    };

    /**
     * @brief Wrapper of GSL multidimensional minimization algorithms without gradient
     *
     * @param[in] n Dimension of the problem
     * @param[in,out] x Initial guess of the solution and the final solution
     * @param[in] func Objective function
     * @param[in] fminimizer Minimization algorithm
     * @param[in] step_size Step size for the algorithm
     * @param[in] size_epsabs Absolute tolerance of the size of the simplex
     * @param[in] max_iter Maximum number of iterations
     * @param[out] fval Final value of the objective function
     */
    void NonGradMultiMin(const int n, double *x, OBJ_FUNC func, const FMINIMIZER fminimizer,
                         const double *step_size, const double size_epsabs, const int max_iter,
                         double &fval);

    enum class ONEMINIMIZER
    {
        goldensection,
        brent,
        quad_golden
    };

    /**
     * @brief Wrapper of GSL one dimensional minimization algorithms
     * The funcion at the lower bound and the upper bound should be larger than the function at the initial guess.
     * @param[in,out] x Initial guess of the solution and the final solution
     * @param[in] func Objective function
     * @param[in] one_minimizer Minimization algorithm
     * @param[in] x_lower Lower bound of the solution
     * @param[in] x_upper Upper bound of the solution
     * @param[in] epsabs Absolute tolerance of the solution
     * @param[in] epsrel Relative tolerance of the solution
     * @param[in] max_iter Maximum number of iterations
     * @param[in] fval Final value of the objective function
     */
    void OneDimMin(double &x, OBJ_ONE_FUNC func, const ONEMINIMIZER oneminimizer,
                   double x_lower, double x_upper, const double epsabs, const double epsrel, const int max_iter,
                   double &fval);

    /**
     * @brief The Tikhonov regularization, C = Sigma * B, where the vectors are all reduced to same length.
     * Applys the bisection-gaurded Newton's method to find the root lambda.
     */
    class VectorizedTikhonov
    {
    public:
        VectorizedTikhonov(const int num = 1);
        ~VectorizedTikhonov() {};

        int size() const;
        void Resize(const int num);

    private:
        void Allocate();

    public:
        void ResetParameters()
        {
            lambda_ = 1.0;
            tol_ = 1e-15;
            max_iter_ = 100;
        }

        /**
         * @brief Calculate the vectorized Tikhonov regularization
         *
         * @param[in] C Data vector
         * @param[in] Sigma Diagonalized coefficient matrix
         * @param[in] tau Parameter of the regularization
         * @param[out] B Solution to the optimization
         */
        void Compute(const double *C, const double *Sigma, const double tau, double *B);

        double lambda_; //!< Strength of the regularization

        double tol_;   //!< Tolerance of the root find algorithm
        int max_iter_; //!< Maximum number of the iteration in root-finding

        std::vector<double> C_squared_;     //!< C^2
        std::vector<double> Sigma_squared_; //!< Sigma^2

    private:
        int num_ = 1; //!< Length of the vectors C, sigma, and B

        /**
         * @brief Calculate the value and its derivative of the objective function of the root-finding problem
         *
         * @param[in] x The root-finding variable
         * @return std::tuple<double, double>, value and derivative of the objective function
         */
        std::tuple<double, double> LHSandDeriv(const double x) const;
    };
} // namespace mtecs3d::utils::optimize