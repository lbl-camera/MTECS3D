#include "mtecs3d.h"
#include "mtecs3d/Common.h"
#include "utils.h"

#include <iostream>

#include "gsl/gsl_sf_legendre.h"
#include "hdf5.h"
#include <Eigen/Dense>

using std::cout, std::endl;

namespace mtecs3d::detail
{

    /**
     * @brief Calculate the matrix of the Legendre polynomials in the expansion of the cross-correlation
     * Running checked.
     * @param[in] lmax maximum reachable values of l
     * @param[in] phi vector of size (num_phi, ), values of phi angles
     * @param[in] theta_q_1 value of the angle associated with q_1 and shape of Ewald sphere
     * @param[in] theta_q_2 value of the angle associated with q_2 and shaoe of Ewald sphere
     * @return MatrixXd, matrix of size (num_phi, lmax/2), the matrix of Legendre polynomials
     */
    Eigen::MatrixXd form_legendre_polynomial_matrix(const int lmax, const Eigen::VectorXd &phi, const double theta_q_1,
                                                    const double theta_q_2)
    {
#ifdef _DEBUG_
        assert(lmax % 2 == 0);
#endif
        const int num_phi = phi.size();
        Eigen::MatrixXd Pmat(num_phi, lmax / 2);
        for (int ind_phi = 0; ind_phi < num_phi; ind_phi++)
        {
            double angles = cos(theta_q_1) * cos(theta_q_2) + sin(theta_q_1) * sin(theta_q_2) * cos(phi[ind_phi]);
            for (int ind_l = 2; ind_l <= lmax; ind_l += 2)
            {
                Pmat(ind_phi, ind_l / 2 - 1) = gsl_sf_legendre_Pl(ind_l, angles) / 4.0 / M_PI;
            }
        }

        Pmat.rowwise() -= Pmat.colwise().mean();

        return Pmat;
    }

    /**
     * @brief Calculate the value of theta(q) given the design of the Ewald sphere
     * Correctness checked.
     * @param[in] q length of the scattered vector
     * @param[in] flat_Ewald_sphere flag indicating if the Ewald sphere is flat
     * @param[in] wavelength wavelength of the beam
     * @return double the angle theta(q)
     */
    double theta_q_Ewald_sphere(const double q, const bool flat_Ewald_sphere, const double wavelength)
    {
        if (flat_Ewald_sphere)
        {
            return M_PI / 2;
        }
        else
        {
            return acos(q * wavelength / 2);
        }
    }
} // namespace mtecs3d::detail

namespace mtecs3d
{
    void ReduceCorrelationData(const char *input_filename, const char *output_filename, const int lmax,
                               const bool flat_Ewald_sphere, const double wavelength, const std::array<int, 2> truncation_limit,
                               const int verbose)
    {
        herr_t ret;
        cout << "Read correlation data from: " << input_filename << endl;
        auto input_file_id = H5Fopen(input_filename, H5F_ACC_RDONLY, H5P_DEFAULT);

        // read correlation from the input file
        auto correlation_dest_id = H5Dopen(input_file_id, "correlation", H5P_DEFAULT);
        auto correlation_space_id = H5Dget_space(correlation_dest_id);
        hsize_t correlation_dims[4];
        H5Sget_simple_extent_dims(correlation_space_id, correlation_dims, nullptr);
        const hsize_t num_delta_t = correlation_dims[0];
        // const hsize_t num_delta_t = 1;
        // correlation_dims[0] = 1;
        const hsize_t num_q = correlation_dims[1];
        const hsize_t num_phi = correlation_dims[3];
        double *correlation = new double[num_delta_t * num_q * num_q * num_phi];
        std::fill_n(correlation, num_delta_t * num_q * num_q * num_phi, 0.0);
        hsize_t correlation_offset[4]{0, 0, 0, 0};
        mtecs3d::utils::data_io::read(input_file_id, "correlation", correlation_offset, correlation_dims, correlation);
        if (verbose >= 1)
        {
            cout << "Correlation data read in" << endl;
        }

        double *measured_q = new double[num_q];
        std::fill_n(measured_q, num_q, 0.0);
        hsize_t q_in_offset[1]{0};
        hsize_t q_in_count[1]{num_q};
        mtecs3d::utils::data_io::read(input_file_id, "q", q_in_offset, q_in_count, measured_q);
        H5Fclose(input_file_id);
        if (verbose >= 1)
        {
            cout << "Measured q grid read in" << endl;
        }

        auto output_file_id = H5Fcreate(output_filename, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
        hsize_t q_out_dims[1]{num_q};
        hsize_t q_out_offset[1]{0};
        mtecs3d::utils::data_io::create_write(output_file_id, "q", 1, q_out_dims, q_out_offset, q_out_dims, measured_q);
        if (verbose >= 1)
        {
            cout << "Measured q grid wrote to output" << endl;
        }

        // calculate the reduced correlation and magunitude of noise
        const hsize_t num_l = lmax / 2;
        double *num_phi_q1q2 = new double[num_delta_t * num_q * num_q];
        std::fill_n(num_phi_q1q2, num_delta_t * num_q * num_q, 0.0);
        double *vt = new double[num_delta_t * num_q * num_q * num_l * num_l];
        std::fill_n(vt, num_delta_t * num_q * num_q * num_l * num_l, 0.0);
        double *eigvals = new double[num_delta_t * num_q * num_q * num_l];
        std::fill_n(eigvals, num_delta_t * num_q * num_q * num_l, 0.0);
        double *reduced_correlation = new double[num_delta_t * num_q * num_q * num_l];
        std::fill_n(reduced_correlation, num_delta_t * num_q * num_q * num_l, 0.0);
        double *squared_noise_magnitude = new double[num_delta_t * num_q * num_q];
        std::fill_n(squared_noise_magnitude, num_delta_t * num_q * num_q, 0.0);

        std::vector<double> superb(num_l);
        for (int ind_t = 0; ind_t < num_delta_t; ind_t++)
        {
            for (int ind_q1 = 0; ind_q1 < num_q; ind_q1++)
            {
                for (int ind_q2 = 0; ind_q2 < num_q; ind_q2++)
                {
                    const int ind_offset = (ind_t * num_q + ind_q1) * num_q + ind_q2;
                    const double theta_q1 = detail::theta_q_Ewald_sphere(measured_q[ind_q1], flat_Ewald_sphere, wavelength);
                    const double theta_q2 = detail::theta_q_Ewald_sphere(measured_q[ind_q2], flat_Ewald_sphere, wavelength);
                    num_phi_q1q2[ind_offset] = truncation_limit[1] - truncation_limit[0];
                    Eigen::VectorXd phi_angles = Eigen::VectorXd::LinSpaced(static_cast<int>(num_phi_q1q2[ind_offset]),
                                                                            truncation_limit[0], truncation_limit[1] - 1);
                    phi_angles.array() *= 2.0 * M_PI / double(num_phi);

                    Eigen::MatrixXd Pmat = detail::form_legendre_polynomial_matrix(lmax, phi_angles, theta_q1, theta_q2);
                    Eigen::MatrixXd U(Pmat.rows(), Pmat.cols()), VT(Pmat.cols(), Pmat.cols());
                    Eigen::VectorXd Pmat_eigvals(Pmat.cols());
                    LAPACKE_dgesvd(LAPACK_COL_MAJOR, 'S', 'S', Pmat.rows(), Pmat.cols(), Pmat.data(), Pmat.rows(), Pmat_eigvals.data(), U.data(), Pmat.rows(), VT.data(), Pmat.cols(), superb.data());
                    cblas_dcopy(num_l * num_l, VT.data(), 1, vt + ind_offset * num_l * num_l, 1);
                    cblas_dcopy(num_l, Pmat_eigvals.data(), 1, eigvals + ind_offset * num_l, 1);

                    Eigen::Map<Eigen::VectorXd> corr_vec(correlation + ind_offset * num_phi + truncation_limit[0],
                                                         num_phi_q1q2[ind_offset]);
                    corr_vec.array() -= corr_vec.mean();
                    Eigen::Map<Eigen::VectorXd> reduced_corr_vec(reduced_correlation + ind_offset * num_l, num_l);
                    reduced_corr_vec = U.transpose() * corr_vec;
                    const double total_norm = corr_vec.norm();
                    const double signal_norm = reduced_corr_vec.norm();
                    squared_noise_magnitude[ind_offset] = total_norm * total_norm - signal_norm * signal_norm;
                }
            }
            if (verbose >= 1)
            {
                cout << "Reduced data of " << ind_t << "-th time slice calculated" << endl;
            }
        }

        delete[] correlation;
        delete[] measured_q;

        hsize_t num_phi_q1q2_dims[3]{num_delta_t, num_q, num_q};
        hsize_t num_phi_q1q2_offset[3]{0, 0, 0};
        mtecs3d::utils::data_io::create_write(output_file_id, "num_phi", 3, num_phi_q1q2_dims, num_phi_q1q2_offset,
                                              num_phi_q1q2_dims, num_phi_q1q2);
        delete[] num_phi_q1q2;

        hsize_t vt_dims[5]{num_delta_t, num_q, num_q, num_l, num_l};
        hsize_t vt_offset[5]{0, 0, 0, 0, 0};
        mtecs3d::utils::data_io::create_write(output_file_id, "vt", 5, vt_dims, vt_offset, vt_dims, vt);
        delete[] vt;

        hsize_t eigvals_dim[4]{num_delta_t, num_q, num_q, num_l};
        hsize_t eigvals_offset[4]{0, 0, 0, 0};
        mtecs3d::utils::data_io::create_write(output_file_id, "eigvals", 4, eigvals_dim, eigvals_offset, eigvals_dim, eigvals);
        delete[] eigvals;

        hsize_t reduced_correlation_dims[4]{num_delta_t, num_q, num_q, num_l};
        hsize_t reduced_correlation_offset[4]{0, 0, 0, 0};
        mtecs3d::utils::data_io::create_write(output_file_id, "reduced_correlation", 4, reduced_correlation_dims,
                                              reduced_correlation_offset, reduced_correlation_dims, reduced_correlation);
        delete[] reduced_correlation;

        hsize_t noise_magnitude_dims[3]{num_delta_t, num_q, num_q};
        hsize_t noise_magnitude_offset[3]{0, 0, 0};
        mtecs3d::utils::data_io::create_write(output_file_id, "squared_noise_magnitude", 3, noise_magnitude_dims,
                                              noise_magnitude_offset, noise_magnitude_dims, squared_noise_magnitude);
        delete[] squared_noise_magnitude;

        H5Fclose(output_file_id);
        cout << "Reduced correlation data wrote to: " << output_filename << endl;
    }
} // namespace mtecs3d