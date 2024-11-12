#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <regex>
#include <string>

#include "mtecs3d.h"
#include "mtecs3d/Common.h"

int Reduce(char *config_file);
int Extract(char *config_file);

int main(int argc, char **argv)
{
    std::cout << std::setprecision(16);
    std::cout << std::endl;
    if (argc != 3)
    {
        std::cerr << "Usage: " << argv[0] << " [reduce|extract] config_file" << std::endl;
        return 1;
    }

    // check if argv[1] is reduce or extract
    if (std::string(argv[1]) == "reduce")
    {
        Reduce(argv[2]);
    }
    else if (std::string(argv[1]) == "extract")
    {
        Extract(argv[2]);
    }
    else if (std::string(argv[1]) == "reduce|extract")
    {
        Reduce(argv[2]);
        Extract(argv[2]);
    }
    else
    {
        std::cerr << "Unknown command: " << argv[1] << std::endl;
        std::cerr << "Usage: " << argv[0] << " [reduce|extract] config_file" << std::endl;
        return 1;
    }

    return 0;
}

int Reduce(char *config_file)
{
        std::ifstream infile(config_file);
    
        std::string correlation_filename;
        int lmax;
        bool flat_Ewald_sphere;
        double wavelength;
        std::array<int, 2> truncation_limit;
        std::string reduced_correlation_filename;
        int verbose = 0;

        if (!infile)
        {
            std::cerr << "Cannot open config file: " << config_file << std::endl;
            return 1;
        }

        std::string line;
        while (std::getline(infile, line))
        {
            if (line.empty() || line[0] == '#')
            {
                continue;
            }
            std::istringstream iss(line);
            std::string key;
            iss >> key;
            if (key == "CorrFile")
            {
                iss >> correlation_filename;
            }
            else if (key == "ReducedCorrFile")
            {
                iss >> reduced_correlation_filename;
            }
            else if (key == "Lmax")
            {
                iss >> lmax;
            }
            else if (key == "FlatEwaldSphere")
            {
                iss >> flat_Ewald_sphere;
            }
            else if (key == "Wavelength")
            {
                iss >> wavelength;
            }
            else if (key == "TruncationLimit")
            {
                iss >> truncation_limit[0] >> truncation_limit[1];
            }
            else if (key == "Verbose")
            {
                iss >> verbose;
            }
        }
        infile.close();

        mtecs3d::ReduceCorrelationData(correlation_filename.c_str(), reduced_correlation_filename.c_str(),
                                       lmax, flat_Ewald_sphere, wavelength, truncation_limit, verbose);

        return 0;
}

int Extract(char *config_file)
{
        std::ifstream infile(config_file);
        
        std::string reduced_correlation_filename;
        int maxIter;
        double tol;
        double diameter;
        double delta_t = 1.0;
        int verbose = 0;

        if (!infile)
        {
            std::cerr << "Cannot open config file: " << config_file << std::endl;
            return 1;
        }

        std::string line;
        while (std::getline(infile, line))
        {
            if (line.empty() || line[0] == '#')
            {
                continue;
            }
            std::istringstream iss(line);
            std::string key;
            iss >> key;
            if (key == "ReducedCorrFile")
            {
                iss >> reduced_correlation_filename;
            }
            else if (key == "DeltaT")
            {
                iss >> delta_t;
            }
            else if (key == "Diameter")
            {
                iss >> diameter;
            }
            else if (key == "MaxExtractIter")
            {
                iss >> maxIter;
            }
            else if (key == "Tol")
            {
                iss >> tol;
            }
            else if (key == "Verbose")
            {
                iss >> verbose;
            }
        }
        infile.close();

        mtecs3d::ExtractCoefficient(reduced_correlation_filename.c_str(), delta_t, diameter, maxIter, tol, verbose);

        return 0;
}
