cmake_minimum_required(VERSION 3.20)
message(STATUS "CMake version: ${CMAKE_VERSION}")

# Set the project name
project(MTECS3D
        VERSION 0.1
        DESCRIPTION "Multi-Tiered Estimation for Correlation Spectroscopy in 3D"
        LANGUAGES C CXX)
message(STATUS "Building Project: ${PROJECT_NAME}")

# message compiler
message(STATUS "CXX compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "CXX compiler ID: ${CMAKE_CXX_COMPILER_ID}")

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Set the output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../bin)

# Specify the source files and include directories
include_directories(${PROJECT_SOURCE_DIR}/src)
include_directories(${PROJECT_SOURCE_DIR}/include)

# Set the compile options
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# Collect all source files in src directory
set(SOURCES CorrelationNoiseProjector.cpp
            BandLimitingProjector.cpp
            TensorDecomp.cpp
            utils.cpp
            optimize.cpp
            ReduceCorr.cpp
            ExtractCoefficient.cpp)
list(TRANSFORM SOURCES PREPEND "${PROJECT_SOURCE_DIR}/src/")        

# Find OpenMP
find_package(OpenMP)
# Find mkl
set(MKL_ARCH intel64)
set(MKL_LINK static)
set(MKL_INTERFACE_FULL intel_lp64)
set(MKL_THREADING sequential)
find_package(MKL)
# check if MKL is found
if(MKL_FOUND)
message(STATUS "MKL found")
else()
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
message(STATUS "MKL not found, use BLAS and LAPACK instead")
endif()
# Find the eigen3
if(MKL_FOUND)
    set(EIGEN_USE_MKL_ALL)
else()
    set(EIGEN_USE_BLAS)
    set(EIGEN_USE_LAPACK)
endif()
find_package(Eigen3 REQUIRED NO_MODULE)
# Find hdf5
find_package(HDF5 REQUIRED)
if(HDF5_IS_PARALLEL)
    message(STATUS "HDF5 is parallel")
    find_package(MPI REQUIRED)
endif()
# Find GSL
find_package(GSL REQUIRED)

# Add the library
add_library(mtecs3d_static STATIC ${SOURCES})
set_target_properties(mtecs3d_static PROPERTIES ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/lib")

# Add the executable
add_executable(mtecs3d ${PROJECT_SOURCE_DIR}/main/mtecs3d.cpp)

function(link_external target)
    # Link eigen3
    target_link_libraries(${target} PUBLIC Eigen3::Eigen)
    # Link MKL or BLAS and LAPACK
    if(MKL_FOUND)
        target_compile_options(${target} PUBLIC $<TARGET_PROPERTY:MKL::MKL,INTERFACE_COMPILE_OPTIONS>)
        target_link_libraries(${target} PUBLIC MKL::MKL)
    else()
        target_link_libraries(${target} PUBLIC BLAS::BLAS LAPACK::LAPACK)
    endif()
    # Link HDF5
    target_link_libraries(${target} PUBLIC hdf5::hdf5)
    if(HDF5_IS_PARALLEL)
        target_link_libraries(${target} PUBLIC MPI::MPI_CXX)
    endif()
    # Link GSL
    target_link_libraries(${target} PUBLIC GSL::gsl GSL::gslcblas)
    # Link OpenMP
    if(OpenMP_CXX_FOUND)
        target_link_libraries(${target} PUBLIC OpenMP::OpenMP_CXX)
    endif()
endfunction()

link_external(mtecs3d_static)
# link_external(mtecs3d)
target_link_libraries(mtecs3d PUBLIC mtecs3d_static)

find_package(Doxygen)
if(DOXYGEN_FOUND)
    set(DOXYGEN_EXCLUDE "${PROJECT_SOURCE_DIR}/build")
    if(DOXYGEN_HAVE_DOT)
        set(DOXYGEN_CALL_GRAPH YES)
        set(DOXYGEN_CALLER_GRAPH YES)
    endif()
    set(DOXYGEN_EXTRACT_ALL YES)
    set(DOXYGEN_EXTRACT_PRIVATE YES)
    set(DOXYGEN_EXTRACT_STATIC YES)
    set(DOXYGEN_GENERATE_TREEVIEW YES)
    set(DOXYGEN_GENERATE_LATEX YES)
    doxygen_add_docs(docs ${PROJECT_SOURCE_DIR})
endif()
