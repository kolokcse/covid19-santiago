cmake_minimum_required (VERSION 3.15)
project(Simulator)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

#include(FetchContent)
#FetchContent_Declare(
#    tensorboard_logger
#    GIT_REPOSITORY      https://github.com/protocolbuffers/protobuf.git
#    GIT_TAG             master
#)
#FetchContent_MakeAvailable(tensorboard_logger)
#
#
#FetchContent_Declare(
#    tensorboard_logger
#    GIT_REPOSITORY      https://github.com/dczifra/tensorboard_logger.git
#    GIT_TAG             master
#)
#FetchContent_MakeAvailable(tensorboard_logger)


set(CMAKE_CXX_FLAGS "-O3")
include_directories(${CMAKE_SOURCE_DIR}/include)
add_executable (main main.cpp include/json.hpp include/sampler.cpp)
target_link_libraries(main)

