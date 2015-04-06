cmake_minimum_required(VERSION 2.6)
project(CUDARayTracer)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_subdirectory(libs/glm)
add_subdirectory(libs/sdl2)
add_subdirectory(libs/tinyobjloader)
add_subdirectory(libs/tinyxml2)

include_directories(
	${SDL2_SOURCE_DIR}/include
	${CMAKE_SOURCE_DIR}/libs/glm
	${CMAKE_SOURCE_DIR}/libs/tinyobjloader
	${CMAKE_SOURCE_DIR}/libs/tinyxml2
	${CMAKE_SOURCE_DIR}
)

add_subdirectory(application)

