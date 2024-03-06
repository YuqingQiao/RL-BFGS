cmake_minimum_required(VERSION 3.10)

project(IK_Controller)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")

# Find the Eigen3 library
find_package(Eigen3 REQUIRED)

# Add the sources to build
add_library(solve_ik SHARED
    src/solve_ik.cpp
    src/forward_kinematics.cpp
    src/capsule_distance.cpp
    src/cma_es.cpp
)
target_link_libraries(solve_ik Eigen3::Eigen)
