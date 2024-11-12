#include "optimize.h"

#include <cmath>
#include <iostream>

#include "gsl/gsl_min.h"
#include "gsl/gsl_multimin.h"

namespace mtecs3d::utils::optimize
{
    struct FDF_PARAMS
    {
        OBJ_FUNC func = nullptr;
    };

    double gsl_func_wrapper(const gsl_vector *x, void *params)
    {
        FDF_PARAMS *func_params = static_cast<FDF_PARAMS *>(params);
        double ret_val = func_params->func(x->data);
        return ret_val;
    }

    void NonGradMultiMin(const int n, double *x, OBJ_FUNC func, const FMINIMIZER fminimizer,
                         const double *step_size, const double size_epsabs, const int max_iter,
                         double &fval)
    {
        int iter = 0;
        int status;
        double size;

        const gsl_multimin_fminimizer_type *T = nullptr;
        gsl_multimin_fminimizer *s = nullptr;

        gsl_vector *x_vec = nullptr;
        gsl_vector *step_size_vec = nullptr;
        gsl_multimin_function my_func;

        FDF_PARAMS fdf_params;
        fdf_params.func = func;
        my_func.n = n;
        my_func.f = &gsl_func_wrapper;
        my_func.params = &fdf_params;

        x_vec = gsl_vector_alloc(n);
        std::copy_n(x, n, x_vec->data);
        step_size_vec = gsl_vector_alloc(n);
        std::copy_n(step_size, n, step_size_vec->data);

        switch (fminimizer)
        {
        case FMINIMIZER::nmsimplex2:
            T = gsl_multimin_fminimizer_nmsimplex2;
            break;

        case FMINIMIZER::nmsimplex:
            T = gsl_multimin_fminimizer_nmsimplex;
            break;

        case FMINIMIZER::nmsimplex2rand:
            T = gsl_multimin_fminimizer_nmsimplex2rand;
            break;

        default:
            break;
        }

        s = gsl_multimin_fminimizer_alloc(T, n);
        gsl_multimin_fminimizer_set(s, &my_func, x_vec, step_size_vec);

        do
        {
            iter++;
            status = gsl_multimin_fminimizer_iterate(s);

            if (status)
            {
                break;
            }

            size = gsl_multimin_fminimizer_size(s);
            status = gsl_multimin_test_size(size, size_epsabs);

            if (status == GSL_SUCCESS)
            {
                break;
            }

        } while (status == GSL_CONTINUE && iter < max_iter);

        std::copy_n(s->x->data, n, x);
        fval = s->fval;

        gsl_multimin_fminimizer_free(s);
        gsl_vector_free(step_size_vec);
        gsl_vector_free(x_vec);
    }

    struct ONE_FDF_PARAMS
    {
        OBJ_ONE_FUNC func = nullptr;
    };

    double gsl_one_func_wrapper(const double x, void *params)
    {
        ONE_FDF_PARAMS *func_params = static_cast<ONE_FDF_PARAMS *>(params);
        double ret_val = func_params->func(x);
        return ret_val;
    }

    void OneDimMin(double &x, OBJ_ONE_FUNC func, const ONEMINIMIZER oneminimizer,
                   double x_lower, double x_upper, const double epsabs, const double epsrel, const int max_iter,
                   double &fval)
    {
        int iter = 0;
        int status;
        const gsl_min_fminimizer_type *T;
        gsl_min_fminimizer *s;
        gsl_function F;
        double x_init = x;

        ONE_FDF_PARAMS params;
        params.func = func;
        F.function = &gsl_one_func_wrapper;
        F.params = &params;

        switch (oneminimizer)
        {
        case ONEMINIMIZER::goldensection:
            T = gsl_min_fminimizer_goldensection;
            break;

        case ONEMINIMIZER::brent:
            T = gsl_min_fminimizer_brent;
            break;

        case ONEMINIMIZER::quad_golden:
            T = gsl_min_fminimizer_quad_golden;
            break;

        default:
            break;
        }
        s = gsl_min_fminimizer_alloc(T);
        gsl_min_fminimizer_set(s, &F, x_init, x_lower, x_upper);

        do
        {
            iter++;
            status = gsl_min_fminimizer_iterate(s);

            if (status)
            {
                break;
            }
            x_lower = gsl_min_fminimizer_x_lower(s);
            x_upper = gsl_min_fminimizer_x_upper(s);

            status = gsl_min_test_interval(x_lower, x_upper, epsabs, epsrel);

            if (status == GSL_SUCCESS)
            {
                break;
            }

        } while (status == GSL_CONTINUE && iter < max_iter);

        x = s->x_minimum;
        fval = s->f_minimum;

        gsl_min_fminimizer_free(s);
    }

    VectorizedTikhonov::VectorizedTikhonov(const int num) : num_{num}
    {
        Allocate();
        std::fill(C_squared_.begin(), C_squared_.end(), 0.0);
        std::fill(Sigma_squared_.begin(), Sigma_squared_.end(), 0.0);
        ResetParameters();
    }

    int VectorizedTikhonov::size() const
    {
        return num_;
    }

    void VectorizedTikhonov::Resize(const int num)
    {
        num_ = num;
        Allocate();
    }

    void VectorizedTikhonov::Allocate()
    {
        C_squared_.resize(num_);
        Sigma_squared_.resize(num_);
    }

    void VectorizedTikhonov::Compute(const double *C, const double *Sigma, const double tau, double *B)
    {
#pragma omp parallel for simd
        for (int ind = 0; ind < num_; ind++)
        {
            C_squared_[ind] = C[ind] * C[ind];
            Sigma_squared_[ind] = Sigma[ind] * Sigma[ind];
        }

        double C_norm = 0;
#pragma omp parallel for schedule(static) reduction(+ : C_norm)
        for (int ind = 0; ind < num_; ind++)
        {
            C_norm += C_squared_[ind];
        }
        if (C_norm <= tau)
        {
#pragma omp parallel for simd
            for (int ind = 0; ind < num_; ind++)
            {
                B[ind] = 0.0;
            }
            lambda_ = 0.0;
            return;
        }

        double lb = 1e-100, ub = 1e100;
        int iter = 0;
        while (true)
        {
            iter++;
            auto [val, dval] = LHSandDeriv(lambda_);
            val -= tau;
            if (std::abs(val) < tau * tol_ || iter > max_iter_)
            {
                break;
            }
            else
            {
                if (val < 0)
                {
                    lb = lambda_;
                }
                else
                {
                    ub = lambda_;
                }
            }

            double next = lambda_ - val / dval;
            if (next > lb && next < ub)
            {
                lambda_ = next;
            }
            else
            {
                if (val > 0)
                {
                    ub = lambda_;
                    if (lb > 0)
                    {

                        lambda_ = (lb + lambda_) / 2;
                    }
                    else
                    {
                        lambda_ /= 10.0;
                    }
                }
                else
                {
                    lb = lambda_;
                    if (ub < 1e50)
                    {
                        lambda_ = (ub + lambda_) / 2;
                    }
                    else
                    {
                        lambda_ *= 10.0;
                    }
                }
            }

            if (lambda_ > 1e100 || lambda_ < 1e-100)
            {
                std::cout << "Bad lambda " << lambda_ << ", val = (" << val << ", " << dval << ")" << std::endl;
                exit(1);
            }
        }

#pragma omp parallel for simd
        for (int ind = 0; ind < num_; ind++)
        {
            B[ind] = C[ind] * Sigma[ind] / (Sigma_squared_[ind] + lambda_);
        }
    }

    std::tuple<double, double> VectorizedTikhonov::LHSandDeriv(const double x) const
    {
        double ret_first = 0, ret_second = 0;
#pragma omp parallel for simd schedule(static) reduction(+ : ret_first, ret_second)
        for (size_t ind = 0; ind < num_; ind++)
        {
            double denominator = x + Sigma_squared_[ind];
            ret_first += std::pow(x / denominator, 2) * C_squared_[ind];
            ret_second += 2 * x * Sigma_squared_[ind] / std::pow(denominator, 3) * C_squared_[ind];
        }
        return std::make_tuple(ret_first, ret_second);
    }
} // namespace mtecs3d::utils::optimize