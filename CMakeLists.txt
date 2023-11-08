cmake_minimum_required(VERSION 3.2)

if (APPLE)
  message(STATUS "Switch to homebrew clang for openmp")
  set(CMAKE_C_COMPILER "/opt/homebrew/opt/llvm/bin/clang")
  set(CMAKE_CXX_COMPILER "/opt/homebrew/opt/llvm/bin/clang++")
endif()

if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
  message( FATAL_ERROR "In-source builds not allowed. Please make a new directory (called a build directory) and run CMake from there. You may need to remove CMakeCache.txt." )
endif()

project(NeuralMetamaterialNetwork CXX)

message(STATUS "${CMAKE_BUILD_TYPE} Build")

# set(CMAKE_BUILD_TYPE Release) 

set(VERSION_MAJOR 0)
set(VERSION_MINOR 1)
set(VERSION_PATCH 0)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/CMake)

set(CMAKE_EXPORT_COMPILE_COMMANDS "ON")
set(CMAKE_CXX_STANDARD 17)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC -O0 -fopenmp")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -O0 -fopenmp -Wno-unused-variable -Wno-unused-but-set-variable -Wsign-compare -Wreorder")
# set(CMAKE_CXX_FLAGS "-Wall -Wextra -Wconversion -Wshadow -Wpedantic -fsanitize=undefined,address")

# libigl
option(LIBIGL_WITH_OPENGL            "Use OpenGL"         ON)
option(LIBIGL_WITH_OPENGL_GLFW       "Use GLFW"           ON)
option(LIBIGL_WITH_OPENGL_GLFW_IMGUI "Use ImGui"                    ON)
option(LIBIGL_WITH_PNG               "Use PNG"                      ON)
option(LIBIGL_WITH_XML               "Use XML"                      ON)


find_package (Eigen3 3.3 REQUIRED NO_MODULE)
find_package (SuiteSparse REQUIRED)
find_package (OpenMP REQUIRED)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

find_package (LIBIGL REQUIRED)

# include(FetchContent)
# FetchContent_Declare(
#     ipc_toolkit
#     GIT_REPOSITORY https://github.com/ipc-sim/ipc-toolkit.git
#     GIT_TAG "main"
# )
# FetchContent_MakeAvailable(ipc_toolkit)


# include(libigl)
# igl_include(glfw)
# igl_include(opengl)
# igl_include(imgui)
# igl_include(copyleft tetgen)
# igl_include(restricted triangle)
# igl_include(stb)


find_package(CGAL REQUIRED COMPONENTS Core)


add_definitions(-DEIGEN_USE_MKL_ALL)
add_definitions(-DUSE_CHOLMOD)
add_definitions(-DMKL_LP64)

include_directories(${CMAKE_SOURCE_DIR})
# include_directories(/opt/intel/oneapi/mkl/2022.1.0/include)
set(SuiteSparse_ROOT ${CMAKE_SOURCE_DIR}/../SuiteSparse-5.12.0)
include_directories(${SuiteSparse_ROOT}/include)

add_subdirectory(Libs)
add_subdirectory(Projects)



message("**************************************************")
message("C++ Cmake Flags: ${CMAKE_CXX_FLAGS}")
message("**************************************************")
