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
		__device__ __host__ inline
		bool intersect(const Ray& r, const HitInfo &info, const Args &args)
		{
			const math::vec3 &invD = args.invD;
			const math::ivec3 &sign = args.sign;

			float tmin, tmax, tymin, tymax, tzmin, tzmax;

			tmin = (bounds[sign.x].x - r.o.x) * invD.x;
			tmax = (bounds[1 - sign.x].x - r.o.x) * invD.x;

			tymin = (bounds[sign.y].y - r.o.y) * invD.y;
			tymax = (bounds[1 - sign.y].y - r.o.y) * invD.y;

			if ((tmin > tymax) || (tymin > tmax))
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

		__device__ __host__ inline
		bool intersect(const Ray& r, const HitInfo &info)
		{
			Args args;
			args.invD = 1.0f / r.d;
			args.sign.x = args.invD.x < 0;
			args.sign.y = args.invD.y < 0;
			args.sign.z = args.invD.z < 0;

			return intersect(r, info, args);
		}
	};

	struct Face
	{
		uint32_t indices[3];

		__host__ __device__ inline
		math::vec3 getCentroid(void *verts)
		{
			Vertex *vs = (Vertex*)verts;
			math::vec3 centroid(0, 0, 0);
			for (int i = 0; i < 3; i++)
			{
				centroid += vs[indices[i]].position;
			}
			return centroid / 3.0f;
		}

		__host__ __device__ inline
		BoundingBox getBoundingBox(void *verts)
		{
			Vertex *vs = (Vertex*)verts;
			
			BoundingBox bb;
			bb.min = math::vec3(FLT_MAX, FLT_MAX, FLT_MAX);
			bb.max = math::vec3(-FLT_MAX, -FLT_MAX, -FLT_MAX);
			for (int i = 0; i < 3; i++)
			{
				bb.min = math::min(bb.min, vs[indices[i]].position);
				bb.max = math::max(bb.max, vs[indices[i]].position);
			}
			return bb;
		}

		__device__ __host__ inline
		bool intersect(const Ray &r, HitInfo &info, const void *verts)
		{
			const Vertex *vertices = (Vertex*)verts;
			const Vertex &v0 = vertices[indices[0]];
			const Vertex &v1 = vertices[indices[1]];
			const Vertex &v2 = vertices[indices[2]];

			const math::vec3 &a = v0.position;
			const math::vec3 &b = v1.position;
			const math::vec3 &c = v2.position;
			math::vec3 bCoords;
			float t;

			if (math::myIntersectRayTriangle(r.o, r.d, a, b, c, bCoords, t) && t < info.t)
			{
				info.t = t;
				info.point.position = r.o + r.d*t;
				info.point.normal = bCoords.x * v0.normal + bCoords.y * v1.normal + bCoords.z * v2.normal;
				info.point.color = bCoords.x * v0.color + bCoords.y * v1.color + bCoords.z * v2.color;
				return true;
			}
			return false;
		}
	};

} // namespace acr

#endif //_GEOMETRY_H_
