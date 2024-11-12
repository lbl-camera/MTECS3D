#include "mtecs3d.h"
#include "mtecs3d/BandLimitingProjector.h"
#include "mtecs3d/Common.h"
#include "mtecs3d/CorrelationNoiseProjector.h"
#include "mtecs3d/TensorDecomp.h"

#include <iostream>

#include "utils.h"

using namespace Eigen;
using std::cout, std::endl;

namespace mtecs3d
{
    void ExtractCoefficient(const char *reduced_correlation_filename, const double delta_t, const double diameter,
                            const int max_mtecs_iter, const double tol_mtecs, const int verbose)
    {
        detail::Dim dim;
        if (verbose >= 1)
        {
            cout << "Initialize correlation noise projector" << endl;
        }
        cout << "Reading reduced_correlation from: " << reduced_correlation_filename << endl;
        detail::CorNoiseProj cor_noise_proj(reduced_correlation_filename, &dim, verbose);
        Tensor<double, 4> B(dim.num_l, dim.num_q, dim.num_q, dim.num_delta_t);
        B.setConstant(0.0);

        // Initialize band-limiting projector
        if (verbose >= 1)
        {
            cout << "Initialize band-limiting projector" << endl;
        }
        detail::BandLimitingProj band_limiting_proj(diameter, &dim);
        detail::TensorVec G(dim.num_l);
        for (int ind_l = 0; ind_l < dim.num_l; ind_l++)
        {
            G[ind_l].resize(dim.num_basis[ind_l], dim.num_basis[ind_l], dim.num_delta_t);
            G[ind_l].setConstant(0.0);
        }

        if (verbose >= 1)
        {
            cout << "Initialize tensor decomposition" << endl;
        }
        detail::TensorIsoRotProj tensor_rot_proj(delta_t, &dim);
        double D_rot = 0.0;

        int iter = 0;
        mtecs3d::utils::Tracker tracker(B);
        mtecs3d::utils::Timer timer;

        Tensor<double, 4> B_buffer = B;
        double rel_diff_B_truth = 1.0;
        double rel_diff_B_prev = 1.0;
        std::vector<double> rel_diff_G_truth(dim.num_l);

        if (verbose >= 1)
        {
            cout << "MTECS iterations started" << endl;
            cout << "------------------------------------" << endl;
        }

        while (iter < max_mtecs_iter && rel_diff_B_prev > tol_mtecs)
        {
            iter++;
            cor_noise_proj.Compute(B);
            band_limiting_proj.ComputeDownward(B, G);
            tensor_rot_proj.Compute(G, D_rot);

            band_limiting_proj.ComputeUpward(G, B);

            rel_diff_B_prev = tracker.Update(B, true);

            timer.Time();
            if (verbose >= 1)
            {
                cout << iter << "-th iteration, "
                     << "time taken: "
                     << timer.Total() << ", "
                     << "diff_B_prev = " << rel_diff_B_prev << ", "
                     << "D_rot = " << D_rot
                     << endl;
            }
        }

        timer.Time();
        if (verbose >= 1)
        {
            cout << "------------------------------------" << endl;
            cout << "Total time taken: " << timer.Total() << endl;
        }

        cout << "D_rot = " << D_rot << endl;
    }
} // namespace mtecs3d