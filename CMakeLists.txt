cmake_minimum_required(VERSION 3.27...3.30)
project(cobweb LANGUAGES CXX)

set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)
find_package(Eigen3 REQUIRED)

pybind11_add_module(cobweb_discrete src/cobweb_discrete.cpp src/helper.cpp
                    WITH_SOABI)
target_include_directories(cobweb_discrete PRIVATE include/)
target_link_libraries(cobweb_discrete PRIVATE pybind11::headers)
target_link_libraries(cobweb_discrete PRIVATE Eigen3::Eigen)
target_compile_features(cobweb_discrete PRIVATE cxx_std_17)
install(TARGETS cobweb_discrete DESTINATION cobweb)

pybind11_add_module(
  cobweb_continuous src/cobweb_continuous.cpp src/cobweb_continuous_node.cpp
  src/cobweb_continuous_tree.cpp src/helper.cpp WITH_SOABI)
target_include_directories(cobweb_continuous PRIVATE include/)
target_link_libraries(cobweb_continuous PRIVATE pybind11::headers)
target_link_libraries(cobweb_continuous PRIVATE Eigen3::Eigen)
target_compile_features(cobweb_continuous PRIVATE cxx_std_17)
install(TARGETS cobweb_continuous DESTINATION cobweb)
