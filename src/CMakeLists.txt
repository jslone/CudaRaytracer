cmake_minimum_required(VERSION 2.6)
project(CUDARayTracer)

set( CMAKE_EXPORT_COMPILE_COMMANDS 1 )
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-std=c++11 -Wall -Wextra -O3")

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_subdirectory(libs/glm)
add_subdirectory(libs/sdl2)
add_subdirectory(libs/assimp)

include_directories(
  ${CMAKE_SOURCE_DIR}/libs/sdl2/include
	${CMAKE_SOURCE_DIR}/libs/glm/glm
  ${CMAKE_SOURCE_DIR}/libs/assimp/include
	${CMAKE_SOURCE_DIR}
)

link_libraries(
  SDL2
  assimp
)

find_package(CUDA QUIET REQUIRED)

set(CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-std=c++11 -O3 -gencode arch=compute_30,code=sm_30")

include_directories(${CUDA_INCLUDE_DIRS})

add_subdirectory(geometry)
link_libraries(geometry)

add_subdirectory(scene)
link_libraries(scene)

add_subdirectory(
  renderer
)

link_libraries(
  renderer
)

add_subdirectory(application)
