#include "mtecs3d/CorrelationNoiseProjector.h"

#include "mtecs3d/Common.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>

#include "gsl/gsl_sf_legendre.h"
#include "hdf5.h"

using std::cout, std::endl;
using namespace Eigen;

namespace mtecs3d::detail
{
    CorNoiseProj::CorNoiseProj(const char *input_filename, Dim *dim_in, const int verbose)
    {
        dim = dim_in;

        auto input_file_id = H5Fopen(input_filename, H5F_ACC_RDONLY, H5P_DEFAULT);
        auto corr_dest_id = H5Dopen(input_file_id, "reduced_correlation", H5P_DEFAULT);
        auto corr_space_id = H5Dget_space(corr_dest_id);
        hsize_t corr_dims[4];
        H5Sget_simple_extent_dims(corr_space_id, corr_dims, nullptr);
        dim->num_delta_t = corr_dims[0];
        dim->num_q = corr_dims[1];
        dim->num_l = corr_dims[3];
        if (verbose >= 1)
        {
            cout << "num_delta_t = " << dim->num_delta_t << ", num_q = " << dim->num_q << ", num_l = " << dim->num_l << endl;
        }
        H5Sclose(corr_space_id);
        H5Dclose(corr_dest_id);

        hsize_t q_offset[1]{0};
        hsize_t q_count[1]{static_cast<hsize_t>(dim->num_q)};
        dim->measured_q.resize(dim->num_q);
        mtecs3d::utils::data_io::read(input_file_id, "q", q_offset, q_count, dim->measured_q.data());
        if (verbose >= 1)
        {
            cout << "measured q read in" << endl;
        }

        hsize_t num_phi_offset[3]{0, 0, 0};
        hsize_t num_phi_count[3]{static_cast<hsize_t>(dim->num_delta_t), static_cast<hsize_t>(dim->num_q),
                                 static_cast<hsize_t>(dim->num_q)};
        Tensor<double, 3> num_phi(dim->num_q, dim->num_q, dim->num_delta_t);
        mtecs3d::utils::data_io::read(input_file_id, "num_phi", num_phi_offset, num_phi_count, num_phi.data());
        if (verbose >= 1)
        {
            cout << "num_phi read in" << endl;
        }

        hsize_t vt_offset[5]{0, 0, 0, 0, 0};
        hsize_t vt_count[5]{static_cast<hsize_t>(dim->num_delta_t), static_cast<hsize_t>(dim->num_q), static_cast<hsize_t>(dim->num_q),
                            static_cast<hsize_t>(dim->num_l), static_cast<hsize_t>(dim->num_l)};
        vt.resize(dim->num_l, dim->num_l, dim->num_q, dim->num_q, dim->num_delta_t);
        mtecs3d::utils::data_io::read(input_file_id, "vt", vt_offset, vt_count, vt.data());
        if (verbose >= 1)
        {
            cout << "vt read in" << endl;
        }

        hsize_t eigvals_offset[4]{0, 0, 0, 0};
        hsize_t eigvals_count[4]{static_cast<hsize_t>(dim->num_delta_t), static_cast<hsize_t>(dim->num_q),
                                 static_cast<hsize_t>(dim->num_q), static_cast<hsize_t>(dim->num_l)};
        eigvals.resize(dim->num_l, dim->num_q, dim->num_q, dim->num_delta_t);
        mtecs3d::utils::data_io::read(input_file_id, "eigvals", eigvals_offset, eigvals_count, eigvals.data());
        if (verbose >= 1)
        {
            cout << "eigvals read in" << endl;
        }

        hsize_t reduced_correlation_offset[4]{0, 0, 0, 0};
        hsize_t reduced_correlation_count[4]{static_cast<hsize_t>(dim->num_delta_t), static_cast<hsize_t>(dim->num_q),
                                             static_cast<hsize_t>(dim->num_q), static_cast<hsize_t>(dim->num_l)};
        reduced_corr.resize(dim->num_l, dim->num_q, dim->num_q, dim->num_delta_t);
        mtecs3d::utils::data_io::read(input_file_id, "reduced_correlation", reduced_correlation_offset,
                                      reduced_correlation_count, reduced_corr.data());
        if (verbose >= 1)
        {
            cout << "reduced_correlation read in" << endl;
        }

        hsize_t squared_noise_magnitude_offset[3]{0, 0, 0};
        hsize_t squared_noise_magnitude_count[3]{static_cast<hsize_t>(dim->num_delta_t), static_cast<hsize_t>(dim->num_q),
                                                 static_cast<hsize_t>(dim->num_q)};
        Tensor<double, 3> squared_noise_magnitude(dim->num_q, dim->num_q, dim->num_delta_t);
        mtecs3d::utils::data_io::read(input_file_id, "squared_noise_magnitude", squared_noise_magnitude_offset,
                                      squared_noise_magnitude_count, squared_noise_magnitude.data());
        if (verbose >= 1)
        {
            cout << "squared_noise_magnitude read in" << endl;
        }

        H5Fclose(input_file_id);

        double Ntot = 0;
        double invNcutavg = 0;
        double invNcutMlnumM2 = 0;
        int valid_q_pair = 0;
        for (int ind_t = 0; ind_t < dim->num_delta_t; ind_t++)
        {
            for (int ind_q1 = 0; ind_q1 < dim->num_q; ind_q1++)
            {
                for (int ind_q2 = 0; ind_q2 < dim->num_q; ind_q2++)
                {
                    double Ncut = num_phi(ind_q2, ind_q1, ind_t);
                    Ntot += Ncut;
                    invNcutavg += 1.0 / Ncut;
                    invNcutMlnumM2 += 1.0 / (Ncut - dim->num_l - 2);
                    valid_q_pair++;
                }
            }
        }
        if (verbose >= 1)
        {
            cout << "parameters for tau calculated" << endl;
        }

        for (int ind_t = 0; ind_t < dim->num_delta_t; ind_t++)
        {
            for (int ind_q1 = 0; ind_q1 < dim->num_q; ind_q1++)
            {
                for (int ind_q2 = 0; ind_q2 < dim->num_q; ind_q2++)
                {
                    double var = squared_noise_magnitude(ind_q2, ind_q1, ind_t) / (num_phi(ind_q2, ind_q1, ind_t) - dim->num_l);
                    var = sqrt(var);
                    for (int ind_l = 0; ind_l < dim->num_l; ind_l++)
                    {
                        eigvals(ind_l, ind_q2, ind_q1, ind_t) /= var * sqrt(Ntot) * dim->measured_q[ind_q1] * dim->measured_q[ind_q2];
                        reduced_corr(ind_l, ind_q2, ind_q1, ind_t) /= var * sqrt(Ntot);
                    }
                }
            }
        }

        if (verbose >= 1)
        {
            cout << "eigvals & reduced_correlation normalized" << endl;
        }

        double eps1 = 1.0;
        double eps2 = double(Ntot - valid_q_pair * dim->num_l) / double(Ntot);
        double invtot = valid_q_pair;
        double errfacden = cblas_dnrm2(dim->num_delta_t * dim->num_q * dim->num_q * dim->num_l, reduced_corr.data(), 1);
        errfacden = Ntot * errfacden * errfacden;
        double errfacnum = valid_q_pair * dim->num_l;

        double errfac = (errfacnum / errfacden - 1) * dim->num_l * invNcutavg / invtot + 1;
        errfac *= 1 + 2 * invNcutMlnumM2 / invtot;
        tau = eps1 * errfac - eps2;

        double snr = cblas_dnrm2(reduced_corr.size(), reduced_corr.data(), 1);
        snr = snr * snr - dim->num_l * valid_q_pair / Ntot;
        if (verbose >= 1)
        {
            cout << "snr = " << snr << endl;
            cout << "tau = " << tau << endl;
            cout << "xi = " << tau + eps2 << endl;
        }

        dim->snr_t.resize(dim->num_delta_t);
        for (int ind_t = 0; ind_t < dim->num_delta_t; ind_t++)
        {
            double val = cblas_dnrm2(dim->num_q * dim->num_q * dim->num_l, reduced_corr.data() + ind_t * dim->num_q * dim->num_q * dim->num_l, 1);
            dim->snr_t[ind_t] = val * val * dim->num_delta_t - dim->num_l * valid_q_pair / Ntot;
        }

        solver.Resize(dim->num_delta_t * dim->num_q * dim->num_q * dim->num_l);
        SetParameters();
        C.resize(dim->num_l, dim->num_q, dim->num_q, dim->num_delta_t);
        delta_B.resize(dim->num_l, dim->num_q, dim->num_q, dim->num_delta_t);
    }

    void CorNoiseProj::Compute(Tensor<double, 4> &B)
    {
        const int num_delta_t = dim->num_delta_t;
        const int num_q = dim->num_q;
        const int num_l = dim->num_l;
#pragma omp parallel for collapse(3)
        for (int ind_t = 0; ind_t < num_delta_t; ind_t++)
        {
            for (int ind_q1 = 0; ind_q1 < num_q; ind_q1++)
            {
                for (int ind_q2 = 0; ind_q2 < num_q; ind_q2++)
                {
                    const int ind_offset = (ind_t * num_q + ind_q1) * num_q + ind_q2;
                    Map<const MatrixXd> vt_mat(vt.data() + ind_offset * num_l * num_l, num_l, num_l);
                    Map<const VectorXd> B_vec(B.data() + ind_offset * num_l, num_l);
                    Map<const VectorXd> reduced_corr_vec(reduced_corr.data() + ind_offset * num_l, num_l);
                    Map<const VectorXd> eigvals_vec(eigvals.data() + ind_offset * num_l, num_l);
                    Map<VectorXd> C_vec(C.data() + ind_offset * num_l, num_l);
                    C_vec = reduced_corr_vec - eigvals_vec.cwiseProduct(vt_mat * B_vec);
                }
            }
        }

        solver.Compute(C.data(), eigvals.data(), tau, delta_B.data());

#pragma omp parallel for collapse(3)
        for (int ind_t = 0; ind_t < num_delta_t; ind_t++)
        {
            for (int ind_q1 = 0; ind_q1 < num_q; ind_q1++)
            {
                for (int ind_q2 = 0; ind_q2 < num_q; ind_q2++)
                {
                    const int ind_offset = (ind_t * num_q + ind_q1) * num_q + ind_q2;
                    Map<const MatrixXd> vt_mat(vt.data() + ind_offset * num_l * num_l, num_l, num_l);
                    Map<VectorXd> delta_B_vec(delta_B.data() + ind_offset * num_l, num_l);
                    Map<VectorXd> B_vec(B.data() + ind_offset * num_l, num_l);
                    B_vec += vt_mat.transpose() * delta_B_vec;
                }
            }
        }
    }
} // namespace mtecs3d::detail