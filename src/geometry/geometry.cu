#include <cstdlib>
#include <cuda.h>
#include "geometry.h"

namespace acr
{
	Mesh::Mesh() {}

	Mesh::Mesh(float *positions, float *normals, float *colors, uint32_t *indices, uint32_t numVertices, uint32_t numFaces)
		: vertices(numVertices)
		, faces(numFaces)
	{

		for (uint32_t i = 0; i < numVertices; i++)
		{
			for (uint32_t j = 0; j < 3; j++)
			{
				vertices[i].position[j] = positions[3 * i + j];
				vertices[i].normal[j] = normals[3 * i + j];
				vertices[i].color[j] = colors ? colors[4 * i + j] : 1.0f;
			}
		}

		for (uint32_t i = 0; i < numFaces; i++)
		{
			for (uint32_t j = 0; j < 3; j++)
			{
				faces[i].indices[j] = indices[3 * i + j];
			}
		}
	}

	bool Mesh::intersect(const Ray &r, HitInfo &info)
	{
		bool intersected = false;
		for (uint32_t i = 0; i < vertices.size(); i++)
		{
			const Vertex &v0 = vertices[faces[i].indices[0]];
			const Vertex &v1 = vertices[faces[i].indices[1]];
			const Vertex &v2 = vertices[faces[i].indices[2]];

			const math::vec3 &a = v0.position;
			math::vec3 b = v1.position - a;
			math::vec3 c = v2.position - a;
			math::vec3 bCoords;

			if (math::intersectRayTriangle(r.o, r.d, a, b, c, bCoords))
			{
				math::vec3 position = bCoords.x*a + bCoords.y*b + bCoords.z*c;

				float t = math::length(position - r.o);

				if (t < info.t)
				{
					intersected = true;

					info.t = t;
					info.point.position = position;
					info.point.normal = bCoords.x * v0.normal + bCoords.y * v1.normal + bCoords.z * v2.normal;
					info.point.color = bCoords.x * v0.color + bCoords.y * v1.color + bCoords.z * v2.color;
					info.materialIndex = materialIndex;
				}
			}
		}
		return intersected;
	}

	void Mesh::flushToDevice()
	{
		vertices.flushToDevice();
		faces.flushToDevice();
	}

} // namespace acr
