#include "utils.h"

#if __has_include("mkl.h")
#include "mkl_cblas.h"
#include "mkl_lapacke.h"
#elif __has_include("cblas.h") && __has_include("lapacke.h")
#include "cblas.h"
#include "lapacke.h"
#else
#error "Missing implementation of CBLAS and LAPACKE"
#endif

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include <Eigen/Dense>

using namespace Eigen;
using std::cout, std::endl;
using namespace std::chrono;

namespace mtecs3d::utils::data_io
{
    herr_t read(const hid_t input_file_id, const char *dest_name, const hsize_t *offset, const hsize_t *count,
                double *buffer)
    {
        herr_t ret;
        auto dest_id = H5Dopen(input_file_id, dest_name, H5P_DEFAULT);
        auto space_id = H5Dget_space(dest_id);
        H5Sselect_hyperslab(space_id, H5S_SELECT_SET, offset, NULL, count, NULL);
        const int rank = H5Sget_simple_extent_ndims(space_id);
        auto mem_id = H5Screate_simple(rank, count, NULL);
        H5Sselect_all(mem_id);
        ret = H5Dread(dest_id, H5T_NATIVE_DOUBLE, mem_id, space_id, H5P_DEFAULT, buffer);
        H5Sclose(mem_id);
        H5Sclose(space_id);
        H5Dclose(dest_id);
        return ret;
    }

    herr_t create_write(const hid_t output_file_id, const char *dest_name, const int rank, const hsize_t *dims,
                        const hsize_t *offset, const hsize_t *count, const double *buffer)
    {
        herr_t ret;
        auto space_id = H5Screate_simple(rank, dims, NULL);
        auto dest_id =
            H5Dcreate(output_file_id, dest_name, H5T_NATIVE_DOUBLE, space_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Sselect_hyperslab(space_id, H5S_SELECT_SET, offset, NULL, count, NULL);
        auto mem_id = H5Screate_simple(rank, count, NULL);
        H5Sselect_all(mem_id);
        ret = H5Dwrite(dest_id, H5T_NATIVE_DOUBLE, mem_id, space_id, H5P_DEFAULT, buffer);
        H5Sclose(mem_id);
        H5Sclose(space_id);
        H5Dclose(dest_id);
        return ret;
    }
} // namespace mtecs3d::utils::data_io

namespace mtecs3d::utils::linalg
{
    std::tuple<MatrixXd, VectorXd, MatrixXd> SingValDecomp(const MatrixXd &A, const SVDTYPE type)
    {
        const int m = A.rows(), n = A.cols();
        const int k = std::min(m, n);

        MatrixXd U, VT;
        int ldvt = k;
        char jobu = 'N', jobvt = 'N';
        if (type == SVDTYPE::ThinSVD)
        {
            jobu = 'S';
            U.resize(m, k);
            jobvt = 'S';
            VT.resize(k, n);
        }
        else if (type == SVDTYPE::FullSVD)
        {
            jobu = 'A';
            U.resize(m, m);
            jobvt = 'A';
            VT.resize(n, n);
            ldvt = n;
        }

        VectorXd s(k);
        std::vector<double> superb(k);
        MatrixXd A_copy = A;
        LAPACKE_dgesvd(LAPACK_COL_MAJOR, jobu, jobvt, m, n, A_copy.data(), m, s.data(), U.data(), m, VT.data(), ldvt, superb.data());

        return std::make_tuple(U, s, VT);
    }
} // namespace mtecs3d::utils::linalg

using namespace std::chrono;
namespace mtecs3d::utils
{
    Timer::Timer()
    {
        begin = high_resolution_clock::now();
        last = begin;
    }

    void Timer::Time()
    {
        const auto current = high_resolution_clock::now();

        const auto total_duration = duration_cast<std::chrono::milliseconds>(current - begin);
        total_time = FormatTimeInMilli(total_duration.count());

        const auto elapsed_duration = duration_cast<std::chrono::milliseconds>(current - last);
        elapsed_time = FormatTimeInMilli(elapsed_duration.count());

        last = current;
    }

    const std::string &Timer::Total() const
    {
        return total_time;
    }

    const std::string &Timer::Elapsed() const
    {
        return elapsed_time;
    }

    std::string FormatTimeInMilli(long long count)
    {
        std::string ret_str = "";

        const long long num_days = count / (86400 * 1000);
        count %= 86400 * 1000;
        if (num_days > 0)
        {
            ret_str += std::to_string(num_days) + "d";
        }

        const long long num_hours = count / (3600 * 1000);
        count %= 3600 * 1000;
        if (num_hours > 0)
        {
            ret_str += std::to_string(num_hours) + "h";
        }

        const long long num_mins = count / (60 * 1000);
        count %= 60 * 1000;
        if (num_mins > 0)
        {
            ret_str += std::to_string(num_mins) + "m";
        }

        const long long num_secs = count / 1000;
        count %= 1000;
        if (num_secs > 0)
        {
            ret_str += std::to_string(num_secs) + "s";
        }

        if (count > 0 || ret_str.empty())
        {
            ret_str += std::to_string(count) + "ms";
        }

        return ret_str;
    }

    const auto start = high_resolution_clock::now();

    std::ostream &Current(std::ostream &os)
    {
        const auto now = system_clock::now();
        const auto now_c = system_clock::to_time_t(now);
        std::string time_str = std::ctime(&now_c);
        time_str.pop_back();

        // Get the elapsed time since start
        const auto elapsed = high_resolution_clock::now() - start;
        const auto elapsed_duration = duration_cast<seconds>(elapsed);
        const auto elapsed_seconds = elapsed_duration.count();
        // Turn the elapsed time into a string in the format "HH:MM:SS"
        const auto hours = elapsed_seconds / 3600;
        std::string hours_str = std::to_string(hours);
        if (hours_str.size() == 1)
        {
            hours_str = "0" + hours_str;
        }
        const auto minutes = (elapsed_seconds % 3600) / 60;
        std::string minutes_str = std::to_string(minutes);
        if (minutes_str.size() == 1)
        {
            minutes_str = "0" + minutes_str;
        }
        const auto seconds = elapsed_seconds % 60;
        std::string seconds_str = std::to_string(seconds);
        if (seconds_str.size() == 1)
        {
            seconds_str = "0" + seconds_str;
        }
        std::string elapsed_str = "Elapsed: " + hours_str + ":" + minutes_str + ":" + seconds_str;
        os << time_str << " [" << elapsed_str << "] ";
        return os;
    }
} // namespace mtecs3d::utils