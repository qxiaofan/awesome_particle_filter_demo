cmake_minimum_required(VERSION 2.8)

project(ParticleFilter)

SET(CMAKE_BUILD_TYPE Debug)

set(CMAKE_CXX_FLAGS "-std=c++11")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -pthread -O0")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -O0")

find_package(OpenCV 3.0 QUIET)
#include_directories( ${OpenCV_INCLUDE_DIR} )

include_directories(
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/include
        ${OpenCV_INCLUDE_DIRS}
)
message(${OpenCV_INCLUDE_DIRS})

add_executable(PF main.cpp
        src/readData.cpp
        src/utils.cpp
        src/Condensation.cpp
        )

target_link_libraries(PF ${OpenCV_LIBS} )
