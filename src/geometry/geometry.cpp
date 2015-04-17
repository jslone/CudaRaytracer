#include <cstdlib>
#include "geometry.h"

namespace acr {

  Mesh::Mesh(float *positions, float *normals, float *colors, uint32_t *indices, uint32_t numVertices, uint32_t numFaces) {
    data = (char*)std::malloc(numVertices * sizeof(Vertex) + numFaces * sizeof(Face));
    vertices = (Vertex*)data;
    faces = (Face*)(data + numVertices * sizeof(Vertex));

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
  }

  Mesh::~Mesh() {
    std::free(data);
  }
} // namespace acr
