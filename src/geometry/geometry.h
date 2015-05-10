#ifndef _GEOMETRY_H_
#define _GEOMETRY_H_

#include "math/math.h"
#include "materials/materials.h"

namespace acr
{

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

	struct BoundingBox
	{
		struct Args
		{
			math::vec3 invD;
			math::ivec3 sign;
		};

		union
		{
			struct { math::vec3 min, max; };
			struct { math::vec3 bounds[2]; };
		};

		// http://people.csail.mit.edu/amy/papers/box-jgt.pdf
		bool intersect(const Ray& r, HitInfo &info, const Args &args)
		{
			const math::vec3 &invD = args.invD;
			const math::ivec3 &sign = args.sign;

			float tmin, tmax, tymin, tymax, tzmin, tzmax;

			tmin = (bounds[sign.x].x - r.o.x) * invD.x;
			tmax = (bounds[1 - sign.x].x - r.o.x) * invD.x;

			tymin = (bounds[sign.y].y - r.o.y) * invD.y;
			tymax = (bounds[1 - sign.y].y - r.o.y) * invD.y;

			if ((tmin > tymin) || (tymin > tmax))
				return false;
			if (tymin > tmin)
				tmin = tymin;
			if (tymax < tmax)
				tmax = tymax;

			tzmin = (bounds[sign.z].z - r.o.z) * invD.z;
			tzmax = (bounds[1 - sign.z].z - r.o.z) * invD.z;

			if ((tmin > tzmax) || (tzmin > tmax))
				return false;
			if (tzmin > tmin)
				tmin = tzmin;
			if (tzmax < tmax)
				tmax = tzmax;

			return tmin < info.t && tmax > math::epsilon<float>();
		}
	};

} // namespace acr

#endif //_GEOMETRY_H_
