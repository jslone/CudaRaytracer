cmake_minimum_required(VERSION 2.6)
project(CUDARayTracer)

set( CMAKE_EXPORT_COMPILE_COMMANDS 1 )
if(CMAKE_COMPILER_IS_GNUCXX)
	set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-std=c++11 -Wall -Wextra -O3")
endif(CMAKE_COMPILER_IS_GNUCXX)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_subdirectory(libs/glm)
add_subdirectory(libs/assimp)
add_subdirectory(libs/glew/build/cmake)
add_subdirectory(libs/FreeGLUT/freeglut/freeglut)

include_directories(
	${CMAKE_SOURCE_DIR}/libs/glm/glm
	${CMAKE_SOURCE_DIR}/libs/assimp/include
	${CMAKE_SOURCE_DIR}/libs/glew/include
	freeglut
	${CMAKE_SOURCE_DIR}
)

find_package(OpenGL QUIET REQUIRED)

link_libraries(
	assimp
	glew
	freeglut
	${OPENGL_LIBRARIES}
)

find_package(CUDA QUIET REQUIRED)

set(CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
set(CUDA_SEPARABLE_COMPILATION ON)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-std=c++11")

include_directories(${CUDA_INCLUDE_DIRS})

cuda_add_executable(application
	${CMAKE_SOURCE_DIR}/renderer/renderer.cu
	${CMAKE_SOURCE_DIR}/scene/scene.cu
	${CMAKE_SOURCE_DIR}/geometry/geometry.cu
	${CMAKE_SOURCE_DIR}/materials/materials.cpp
	${CMAKE_SOURCE_DIR}/application/application.cpp
)
