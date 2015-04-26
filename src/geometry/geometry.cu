#include <cstdlib>
#include <cuda.h>
#include "geometry.h"

namespace acr
{
	
	math::vec3 Ray::get_pixel_dir(const Camera &camera, int ni, int nj)
	{

		math::vec3 dir;
		math::vec3 up;
		float AR;

		math::vec3 cR;
		math::vec3 cU;
		float dist;
		math::vec3 pos;
    
		dir = camera.forward;
		up = camera.up;
		AR = camera.aspectRatio;
		cR = math::cross(dir, up);
		cU = math::cross(cR, dir);
		pos = camera.position;
		dist = math::tan(camera.horizontalFOV/2.0);
		
		return math::normalize(dir + dist*(nj*cU + AR*ni*cR));
	}

	__host__
	Mesh::Mesh(const aiMesh *aiMesh)
	{
		thrust::host_vector<Vertex> vs(aiMesh->mNumVertices);
		for (uint32_t i = 0; i < aiMesh->mNumVertices; i++)
		{
			for (uint32_t j = 0; j < 3; j++)
			{
				vs[i].position[j] = aiMesh->mVertices[i][j];
				vs[i].normal[j] = aiMesh->mNormals[i][j];
				vs[i].color[j] = aiMesh->mColors[0] ? aiMesh->mColors[0][i][j] : 1.0f;
			}
		}
		vertices = vector<Vertex>(vs);

		thrust::host_vector<Face> f(aiMesh->mNumFaces);
		for (uint32_t i = 0; i < aiMesh->mNumFaces; i++)
		{
			for (uint32_t j = 0; j < 3; j++)
			{
				f[i].indices[j] = aiMesh->mFaces[i].mIndices[j];
			}
		}
		faces = vector<Face>(f);
	}
	
	__host__
	Mesh::~Mesh() {}

	__host__
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

} // namespace acr
