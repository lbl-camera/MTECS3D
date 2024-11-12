#pragma once

#include <chrono>
#include <tuple>
#include <vector>

#include <Eigen/Eigen>
#include <hdf5.h>

namespace mtecs3d::utils::data_io
{
    /**
     * @brief Read the dataset of a given HDF5 file into buffer. The data-type should be double.
     * Correctness checked.
     * @param[in] input_file_id The HDF5 ID of the input file.
     * @param[in] dest_name Name of the dataset.
     * @param[in] offset Start of the portion of the dataspace.
     * @param[in] count Count starting from the offset.
     * @param[out] buffer Buffer to store the data read in.
     * @return herr_t, flag indicating whether the H5Dread function is properly executed.
     */
    herr_t read(const hid_t input_file_id, const char *dest_name, const hsize_t *offset, const hsize_t *count,
                double *buffer);

    /**
     * @brief Create the dataset in a given HDF5 file and write the data in buffer into that dataset. The data-type should
     * be double. Correctness checked.
     * @param[in] output_file_id HDF5 ID of the output file.
     * @param[in] dest_name Name of the dataset to be created.
     * @param[in] rank Number of dimensions of the dataset.
     * @param[in] dims Size of the created dataset.
     * @param[in] offset Offset of the portion of the dataset to be written.
     * @param[in] count Count starting from the offset, indicating the amount of data to be write.
     * @param[in] buffer Buffer storing the data to be written.
     * @return herr_t, flag indicating whether the H5Dwrite function is properly executed.
     */
    herr_t create_write(const hid_t output_file_id, const char *dest_name, const int rank, const hsize_t *dims,
                        const hsize_t *offset, const hsize_t *count, const double *buffer);
} // namespace mtecs3d::utils::data_io

namespace mtecs3d::utils::linalg
{
    /**
     * @brief Calculate the distance of two consecutively stored data sequences
     *
     * @tparam T1 type of the first data sequence
     * @tparam T2 type of the second data sequence
     * @param[in] vec1 The first sequence
     * @param[in] vec2 The second sequence
     * @return double, the L2 distance between vec1 and vec2
     */
    template <typename T1, typename T2>
    double Dist(const T1 &vec1, const T2 &vec2)
    {
        assert(vec1.size() == vec2.size() && "Dist: vec1.size() does not match vec2.size().");
        double ret_val = 0.0;
#pragma omp parallel for reduction(+ : ret_val)
        for (int ind = 0; ind < vec1.size(); ind++)
        {
            const double ind_val = static_cast<double>(vec1.data()[ind]) - static_cast<double>(vec2.data()[ind]);
            ret_val += ind_val * ind_val;
        }

        return std::sqrt(ret_val);
    }

    /**
     * @brief Calculate the L2 norm of a consecutively stored data sequence
     *
     * @tparam T, type of the sequence
     * @param[in] vec The data sequence
     * @return double, the L2 norm of vec
     */
    template <typename T>
    double Norm(const T &vec)
    {
        double ret_val = 0.0;
#pragma omp parallel for reduction(+ : ret_val)
        for (int ind = 0; ind < vec.size(); ind++)
        {
            double val = static_cast<double>(vec.data()[ind]);
            ret_val += val * val;
        }

        return std::sqrt(ret_val);
    }

    enum class SVDTYPE
    {
        ThinSVD,
        FullSVD,
        SingValOnly
    };

    /**
     * @brief Calculate the singualr value decomposition of a matrix
     *
     * @param[in] A The input matrix
     * @param[in] type If singular vectors are calculated or not
     * @return std::tuple<Eigen::MatrixXd, Eigen::VectorXd, Eigen::MatrixXd>, left singular vectors, singular values in descending order, right singular vectors
     */
    std::tuple<Eigen::MatrixXd, Eigen::VectorXd, Eigen::MatrixXd>
    SingValDecomp(const Eigen::MatrixXd &A, const SVDTYPE type = SVDTYPE::ThinSVD);
} // namespace mtecs3d::utils::linalg

namespace mtecs3d::utils
{
    /**
     * @brief Time the execution of a code section
     */
    class Timer
    {
    public:
        Timer();

        void Time();

        /**
         * @brief Total time consumed
         *
         * @return const std::string&, time consumed
         */
        const std::string &Total() const;
        /**
         * @brief Time consumed since the last call of Elapsed() or Time()
         *
         * @return const std::string&, time consumed
         */
        const std::string &Elapsed() const;

    private:
        std::chrono::time_point<std::chrono::high_resolution_clock> begin; //!< Start time
        std::chrono::time_point<std::chrono::high_resolution_clock> last;  //!< Last time
        std::string total_time;                                            //!< Total time consumed
        std::string elapsed_time;                                          //!< Time consumed since the last call of Elapsed() or Time()
    };

    /**
     * @brief Format the time in millisecond
     *
     * @param[in] count Time in millisecond
     * @return std::string, formatted time in day:hour:minute:second:millisecond
     */
    std::string FormatTimeInMilli(long long count);

    /**
     * @brief A ostream manipulator that returns the current time in brackets
     *
     * @param[in] os
     * @return std::ostream&
     */
    std::ostream &Current(std::ostream &os);

    /**
     * @brief Track the change of a value or a consecutively stored data structure
     *
     * @tparam Val
     */
    template <typename Val>
    class Tracker
    {
    public:
        Tracker(const Val &val)
        {
            val_ = val;
        }

        /**
         * @brief Update the tracked value or the consecutively stored data structure
         *
         * @param[in] new_val New value or data structure
         * @param[in] if_rel Flag indicating if the distance beteen the new and old values is relative
         * @return double, distance between the new and old values
         */
        double Update(const Val &new_val, const bool if_rel = false)
        {
            double ret_val = 0.0;
            ret_val = mtecs3d::utils::linalg::Dist(val_, new_val);
            cblas_dcopy(val_.size(), new_val.data(), 1, val_.data(), 1);
            if (if_rel)
            {
                ret_val /= mtecs3d::utils::linalg::Norm(val_);
            }

            return ret_val;
        }

    private:
        Val val_; // Tracked value or consecutively stored data structure
    };
} // namespace mtecs3d::utils