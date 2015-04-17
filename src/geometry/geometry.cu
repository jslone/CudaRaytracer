#include <cstdlib>
#include <cuda.h>
#include "geometry.h"

namespace acr {
  Mesh::Mesh() {}
  
  Mesh::Mesh(float *positions, float *normals, float *colors, uint32_t *indices, uint32_t numVertices, uint32_t numFaces)
    : numVertices(numVertices), numFaces(numFaces) {
    
    // Allocate on host
    dataSize = numVertices * sizeof(Vertex) + numFaces * sizeof(Face);

    data = (char*)std::malloc(dataSize);
    
    vertices = (Vertex*)data;
    faces = (Face*)(vertices + numVertices);

    for(uint32_t i = 0; i < numVertices; i++) {
      for(uint32_t j = 0; j < 3; j++) {
        vertices[i].position[j] = positions[3*i+j];
        vertices[i].normal[j] = normals[3*i+j];
        vertices[i].color[j] = colors[3*i+j];
      }
    }

    for(uint32_t i = 0; i < numFaces; i++) {
      for(uint32_t j = 0; j < 3; j++) {
        faces[i].indices[j] = indices[3*i+j];
      }
    }

    // Allocate on device
    devSize = sizeof(Mesh) + dataSize;
    cudaMalloc(&devPtr,devSize);

    Mesh devMesh;
    devMesh.numVertices = numVertices;
    devMesh.numFaces = numFaces;
    devMesh.dataSize = dataSize;
    devMesh.data = (char*)(devPtr + 1);
    devMesh.vertices = (Vertex*)devMesh.data;
    devMesh.faces = (Face*)(devMesh.vertices + numVertices);
    
    cudaMemcpy(devPtr, &devMesh, sizeof(Mesh), cudaMemcpyHostToDevice);
    cudaMemcpy(devMesh.data, data, dataSize, cudaMemcpyHostToDevice);
  }

  Mesh::~Mesh() {
    std::free(data);
    cudaFree(devPtr);
  }
} // namespace acr
