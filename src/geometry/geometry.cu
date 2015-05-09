#include <cstdlib>
#include <cuda.h>
#include "geometry.h"

namespace acr
{
	
	Mesh::Mesh(const aiMesh *aiMesh)
		: materialIndex(aiMesh->mMaterialIndex)
	{
		thrust::host_vector<Vertex> vs(aiMesh->mNumVertices);
		std::cout << "Verts[";

		//To get centroid
		math::vec3 sumVertices(0, 0, 0);

		for (uint32_t i = 0; i < aiMesh->mNumVertices; i++)
		{
			for (uint32_t j = 0; j < 3; j++)
			{
				vs[i].position[j] = aiMesh->mVertices[i][j];
				vs[i].normal[j] = aiMesh->mNormals[i][j];
				vs[i].color[j] = aiMesh->mColors[0] ? aiMesh->mColors[0][i][j] : 1.0f;
			}
			sumVertices += vs[i].position;

			std::cout << "Pos: " << math::to_string(vs[i].position) << ", Norm: " << math::to_string(vs[i].normal) << std::endl;
		}

		//Average to get centroid
		localCentroid = sumVertices / (float)aiMesh->mNumVertices;

		std::cout << "\b]" << std::endl;
		vertices = vector<Vertex>(vs);

		std::cout << "Faces[";
		thrust::host_vector<Face> f(aiMesh->mNumFaces);
		for (uint32_t i = 0; i < aiMesh->mNumFaces; i++)
		{
			std::cout << "<";
			for (uint32_t j = 0; j < 3; j++)
			{
				f[i].indices[j] = aiMesh->mFaces[i].mIndices[j];
				std::cout << f[i].indices[j] << ",";
			}
			std::cout << "\b>,";
		}
		std::cout << "\b]" << std::endl;
		faces = vector<Face>(f);
	}
	
	Mesh::~Mesh() {}

	bool Mesh::intersect(const Ray &r, HitInfo &info)
	{
		bool intersected = false;
		for (uint32_t i = 0; i < faces.size(); i++)
		{
			const Vertex &v0 = vertices[faces[i].indices[0]];
			const Vertex &v1 = vertices[faces[i].indices[1]];
			const Vertex &v2 = vertices[faces[i].indices[2]];

			const math::vec3 &a = v0.position;
			const math::vec3 &b = v1.position;
			const math::vec3 &c = v2.position;
			math::vec3 bCoords;
			float t;

			if (math::myIntersectRayTriangle(r.o, r.d, a, b, c, bCoords, t))
			{
				if (t < info.t)
				{
					intersected = true;

					info.t = t;
					info.point.position = r.o + r.d*t;
					info.point.normal = bCoords.x * v0.normal + bCoords.y * v1.normal + bCoords.z * v2.normal;
					info.point.color = bCoords.x * v0.color + bCoords.y * v1.color + bCoords.z * v2.color;
					info.materialIndex = materialIndex;

					//printf("bary: (%f,%f,%f), norm: (%f,%f,%f)\n", bCoords.x, bCoords.y, bCoords.z, info.point.normal.x, info.point.normal.y, info.point.normal.z);
				}
			}
		}
		return intersected;
	}

} // namespace acr
