#include <cstdlib>
#include <cuda.h>
#include "mesh.h"

namespace acr
{
	
	Mesh::Mesh(const aiMesh *aiMesh)
		: materialIndex(aiMesh->mMaterialIndex)
	{
		thrust::host_vector<Vertex> vs(aiMesh->mNumVertices);

		//To get centroid
		math::vec3 sumVertices(0, 0, 0);
		math::vec3 minBound(FLT_MAX, FLT_MAX, FLT_MAX);
		math::vec3 maxBound(-FLT_MAX, -FLT_MAX, -FLT_MAX);

		for (uint32_t i = 0; i < aiMesh->mNumVertices; i++)
		{
			for (uint32_t j = 0; j < 3; j++)
			{
				vs[i].position[j] = aiMesh->mVertices[i][j];
				vs[i].normal[j] = aiMesh->mNormals[i][j];
				vs[i].color[j] = aiMesh->mColors[0] ? aiMesh->mColors[0][i][j] : 1.0f;
			}
			sumVertices += vs[i].position;
			minBound = math::min(minBound, vs[i].position);
			maxBound = math::max(maxBound, vs[i].position);
		}

		boundingBox.min = minBound;
		boundingBox.max = maxBound;

		//Average to get centroid
		centroid = sumVertices / (float)aiMesh->mNumVertices;

		vertices = vector<Vertex>(vs);

		thrust::host_vector<Face> f(aiMesh->mNumFaces);
		for (uint32_t i = 0; i < aiMesh->mNumFaces; i++)
		{
			for (uint32_t j = 0; j < 3; j++)
			{
				f[i].indices[j] = aiMesh->mFaces[i].mIndices[j];
			}
		}
		faces = BIH<Face>(f, boundingBox, &vs[0]);
	}
	
	Mesh::~Mesh() {}

	bool Mesh::intersect(const Ray &r, HitInfo &info)
	{
		if (boundingBox.intersect(r, info) && faces.intersect(r, info, &vertices[0]))
		{
			info.materialIndex = materialIndex;
			return true;
		}
		return false;
	}

} // namespace acr
