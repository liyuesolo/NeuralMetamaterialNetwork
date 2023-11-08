set(DEPS poisson_disk boost_filesystem clipper tbb tbbmalloc tbbmalloc_proxy Eigen3::Eigen igl::core igl::opengl_glfw igl::opengl_glfw_imgui igl::png mkl_intel_lp64 mkl_sequential mkl_core gmp mpfr suitesparseconfig metis cholmod amd camd ccolamd colamd spqr)



file(GLOB HEADERS "include/*.h" "include/autodiff/*.h" "tactile/.*h" "tactile/.*hpp" )
file(GLOB SOURCES "src/*.cpp" "tactile/*.cpp" "src/autodiff/*.cpp")

add_executable(NeuralMetamaterial ${HEADERS} ${SOURCES})
target_link_libraries(NeuralMetamaterial ${CMAKE_SOURCE_DIR}/../gmsh/build/libgmsh.so)  
target_link_libraries(NeuralMetamaterial ${CMAKE_SOURCE_DIR}/../ipc-toolkit/build/libipc_toolkit.a)
target_link_libraries(NeuralMetamaterial ${CMAKE_SOURCE_DIR}/../ipc-toolkit/build/predicates/libpredicates.a)
target_link_libraries(NeuralMetamaterial ${CMAKE_SOURCE_DIR}/../ipc-toolkit/build/_deps/spdlog-build/libspdlog.a)
target_link_libraries(NeuralMetamaterial ${CMAKE_SOURCE_DIR}/../ipc-toolkit/install/collisiondetection/build/libcollisiondetection.a)
target_link_libraries(NeuralMetamaterial ${CMAKE_SOURCE_DIR}/../ipc-toolkit/build/_deps/tight_inclusion-build/libtight_inclusion.a)


target_include_directories(NeuralMetamaterial PUBLIC ../../Libs/spectra/include)
target_include_directories(NeuralMetamaterial PUBLIC ${CMAKE_SOURCE_DIR}/../ipc-toolkit/build/include)
target_include_directories(NeuralMetamaterial PUBLIC ${CMAKE_SOURCE_DIR}/../gmsh/api)  
target_link_libraries(NeuralMetamaterial ${DEPS})

target_include_directories(NeuralMetamaterial PUBLIC ${CMAKE_SOURCE_DIR}/Libs/cyCodeBase)
