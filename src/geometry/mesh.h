#ifndef _MESH_H_
#define _MESH_H_

#include "assimp/mesh.h"
#include "geometry.h"
#include "utils/vector.h"
#include "utils/bih.h"

namespace acr
{
	class Mesh
	{
	public:
		math::vec3 localCentroid;
		BoundingBox boundingBox;

		__host__ __device__
		Mesh() = default;

		__host__
		Mesh(const aiMesh *aiMesh);

		__host__ __device__
		~Mesh();

		__host__ __device__
		bool intersect(const Ray &r, HitInfo &info);
	private:
		vector<Vertex>  vertices;
		vector<Face>    faces;

		uint32_t materialIndex;
	};

} // namespace acr

#endif //_MESH_H_
