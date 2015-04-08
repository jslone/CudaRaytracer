cmake_minimum_required(VERSION 2.6)
project(CUDARayTracer)

set( CMAKE_EXPORT_COMPILE_COMMANDS 1 )

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_subdirectory(libs/glm)
add_subdirectory(libs/sdl2)
add_subdirectory(libs/tinyobjloader)
add_subdirectory(libs/tinyxml2)

include_directories(
  ${CMAKE_SOURCE_DIR}/libs/sdl2/include
	${CMAKE_SOURCE_DIR}/libs/glm/glm
	${CMAKE_SOURCE_DIR}/libs/tinyobjloader
	${CMAKE_SOURCE_DIR}/libs/tinyxml2
	${CMAKE_SOURCE_DIR}
)

link_libraries(
  SDL2
  tinyobjloader
  tinyxml2
)

add_compile_options(-std=c++11 -Wall -Wextra -O3)

add_subdirectory(renderer)

link_libraries(
  renderer
)

add_subdirectory(application)
