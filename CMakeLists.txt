cmake_minimum_required(VERSION 3.27...3.30)
project(cobweb LANGUAGES CXX)
# set(CMAKE_CXX_STANDARD 20)
# set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)
find_package(Eigen3 REQUIRED)

pybind11_add_module(cobweb_discrete src/cobweb_discrete.cpp WITH_SOABI)
target_include_directories(cobweb_discrete PRIVATE include/)
target_link_libraries(cobweb_discrete PRIVATE pybind11::headers)
target_link_libraries(cobweb_discrete PRIVATE Eigen3::Eigen)
target_compile_features(cobweb_discrete PRIVATE cxx_std_20)

install(TARGETS cobweb_discrete DESTINATION cobweb)
