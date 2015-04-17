cmake_minimum_required(VERSION 2.6)
project(CUDARayTracer)

set( CMAKE_EXPORT_COMPILE_COMMANDS 1 )

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

find_package(CUDA QUIET REQUIRED)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}-O3 -gencode arch=compute_20,code=sm_20)

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

add_compile_options(-std=c++11 -Wall -Wextra -O3)

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
