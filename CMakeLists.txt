cmake_minimum_required(VERSION 3.15...3.27)
project(cobweb LANGUAGES CXX)

# CMake 4.x compatibility: allow FetchContent_Populate to avoid CMP0169 error
if(POLICY CMP0169)
  cmake_policy(SET CMP0169 OLD)
endif()

# Nanobind loading
if(CMAKE_VERSION VERSION_LESS 3.18)
    set(DEV_MODULE Development)
else()
    set(DEV_MODULE Development.Module)
endif()

find_package(
    Python 3.8
    COMPONENTS Interpreter ${DEV_MODULE}
    REQUIRED)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE
        Release
        CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"
        "MinSizeRel" "RelWithDebInfo")
endif()

execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

# Loading Eigen
find_package(Eigen3 REQUIRED)

# Add RapidJSON using FetchContent
include(FetchContent)
FetchContent_Declare(
    rapidjson
    GIT_REPOSITORY https://github.com/Tencent/rapidjson.git
    GIT_TAG v1.1.0
    # GIT_TAG master  # Optionally use master, but better to fix to a version.
)
FetchContent_Populate(rapidjson)
set(RAPIDJSON_INCLUDE_DIR ${rapidjson_SOURCE_DIR}/include)

# Add BS::thread_pool using FetchContent
FetchContent_Declare(
    bshoshany_thread_pool
    GIT_REPOSITORY https://github.com/bshoshany/thread-pool.git
    GIT_TAG v5.0.0
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    SYSTEM
)
FetchContent_MakeAvailable(bshoshany_thread_pool)
add_library(bs_thread_pool INTERFACE)
target_include_directories(bs_thread_pool INTERFACE ${bshoshany_thread_pool_SOURCE_DIR}/include)

# Build Cobweb
nanobind_add_module(cobweb_discrete src/cobweb_discrete.cpp src/helper.cpp)
# target_include_directories(cobweb_discrete PRIVATE include/)
target_include_directories(cobweb_discrete PRIVATE include/ ${RAPIDJSON_INCLUDE_DIR})
# target_link_libraries(cobweb_discrete PRIVATE Eigen3::Eigen)
target_link_libraries(cobweb_discrete PRIVATE Eigen3::Eigen bs_thread_pool)
target_compile_features(cobweb_discrete PRIVATE cxx_std_17)
install(TARGETS cobweb_discrete DESTINATION cobweb)

nanobind_add_module(
    cobweb_continuous src/cobweb_continuous.cpp src/cobweb_continuous_node.cpp
    src/cobweb_continuous_tree.cpp src/helper.cpp)
# target_include_directories(cobweb_continuous PRIVATE include/)
target_include_directories(cobweb_continuous PRIVATE include/ ${RAPIDJSON_INCLUDE_DIR})
# target_link_libraries(cobweb_continuous PRIVATE Eigen3::Eigen)
target_link_libraries(cobweb_continuous PRIVATE Eigen3::Eigen bs_thread_pool)
target_compile_features(cobweb_continuous PRIVATE cxx_std_17)
install(TARGETS cobweb_continuous DESTINATION cobweb)
