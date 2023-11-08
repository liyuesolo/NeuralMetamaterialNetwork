
add_library(clipper STATIC clipper/clipper.cpp)
target_include_directories(clipper PUBLIC clipper/clipper.h)
target_link_libraries(clipper ${DEPS})
set(DEPS ${DEPS} clipper)

add_library(poisson_disk STATIC cyCodeBase)
target_include_directories(poisson_disk PUBLIC cyCodeBase)
target_link_libraries(poisson_disk ${DEPS})
set(DEPS ${DEPS} poisson_disk)