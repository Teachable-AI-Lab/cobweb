cmake_minimum_required(VERSION 3.15...3.29)
project(cobweb LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)
find_package(Eigen3 REQUIRED)

pybind11_add_module(cobweb cobweb/cobweb.cpp)
target_include_directories(cobweb PRIVATE cobweb/)
target_link_libraries(cobweb PRIVATE Eigen3::Eigen)

install(TARGETS cobweb DESTINATION .)
