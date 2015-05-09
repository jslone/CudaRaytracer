#ifndef _GEOMETRY_H_
#define _GEOMETRY_H_

#include "math/math.h"
#include "materials/materials.h"

namespace acr
{
	struct BoundingBox
	{
		math::vec3 min, max;
	};

	struct Vertex
	{
		math::vec3 position;
		math::vec3 normal;
		Color3 color;
	};

	struct Face
	{
		uint32_t indices[3];
	};

	struct Ray
	{
		math::vec3 o;
		math::vec3 d;
	};

	struct HitInfo
	{
		float t;
		Vertex point;
		uint32_t materialIndex;
	};

} // namespace acr

#endif //_GEOMETRY_H_
