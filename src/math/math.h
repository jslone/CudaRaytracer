#ifndef _MATH_H_
#define _MATH_H_

#include "glm.hpp"
#include "gtc/matrix_transform.hpp"
#include "gtx/rotate_vector.hpp"

#if __CUDA_ARCH__
namespace glm
{
	template <template <typename, precision> class matType, typename T, precision P>
	GLM_FUNC_DECL const char* to_string(matType<T, P> const & x) { return "NO STRING FOR YOU."; }
}
#else
#include "gtx/string_cast.hpp"
#endif
namespace acr
{
	namespace math
	{
		using namespace glm;

		GLM_FUNC_QUALIFIER
		float fastersin(float x)
		{
			float fouroverpi = 1.2732395447351627f;
			float fouroverpisq = 0.40528473456935109f;
			float q = 0.77633023248007499f;
			union { float f; uint32_t i; } p = { 0.22308510060189463f };

			union { float f; uint32_t i; } vx = { x };
			uint32_t sign = vx.i & 0x80000000;
			vx.i &= 0x7FFFFFFF;

			float qpprox = fouroverpi * x - fouroverpisq * x * vx.f;

			p.i |= sign;

			return qpprox * (q + p.f * qpprox);
		}

		GLM_FUNC_QUALIFIER
		float fastercos(float x)
		{
			float twooverpi = 0.63661977236758134f;
			float p = 0.54641335845679634f;

			union { float f; uint32_t i; } vx = { x };
			vx.i &= 0x7FFFFFFF;

			float qpprox = 1.0f - twooverpi * vx.f;

			return qpprox + p * qpprox * (1.0f - qpprox * qpprox);
		}

		GLM_FUNC_QUALIFIER
		float fastertanfull(float x)
		{
			float twopi = 6.2831853071795865f;
			float invtwopi = 0.15915494309189534f;

			int k = x * invtwopi;
			float half = (x < 0) ? -0.5f : 0.5f;
			float xnew = x - (half + k) * twopi;

			return fastersin(xnew) / fastercos(xnew);
		}


		template<typename genType>
		GLM_FUNC_QUALIFIER genType epsilon()
		{
			return genType(0.001f);
			genType v = genType(1.0f);
			(*((int*)&v))++;
			return v - genType(1.0f);
		}


		// glm: gtx/intersect.inl
		template <typename genType>
		GLM_FUNC_QUALIFIER bool intersectRayTriangle
		(
			genType const & orig, genType const & dir,
			genType const & v0, genType const & v1, genType const & v2,
			genType & baryPosition
		)
		{
			genType e1 = v1 - v0;
			genType e2 = v2 - v0;

			genType p = glm::cross(dir, e2);

			typename genType::value_type a = glm::dot(e1, p);

			typename genType::value_type Epsilon = epsilon<typename genType::value_type>();
			if(a < Epsilon)
				return false;

			typename genType::value_type f = typename genType::value_type(1.0f) / a;

			genType s = orig - v0;
			baryPosition.x = f * glm::dot(s, p);
			if(baryPosition.x < typename genType::value_type(0.0f))
				return false;
			if(baryPosition.x > typename genType::value_type(1.0f))
				return false;

			genType q = glm::cross(s, e1);
			baryPosition.y = f * glm::dot(dir, q);
			if(baryPosition.y < typename genType::value_type(0.0f))
				return false;
			if(baryPosition.y + baryPosition.x > typename genType::value_type(1.0f))
				return false;

			baryPosition.z = f * glm::dot(e2, q);

			return baryPosition.z >= typename genType::value_type(0.0f);
		}

		GLM_FUNC_QUALIFIER bool myIntersectRayTriangle(const vec3 &ro, const vec3 rd, const vec3 &a, const vec3 &b, const vec3 &c, vec3 &baryCoords, float &t)
		{
			vec3 e1, e2;
			vec3 P, Q, T;
			float det, invDet, eps;

			e1 = b - a;
			e2 = c - a;

			P = cross(rd, e2);
			det = dot(e1, P);

			eps = epsilon<float>();

			if (-eps < det && det < eps) // i.e. zero
			{
				return false;
			}

			invDet = 1.0f / det;

			T = ro - a;

			baryCoords.y = dot(T, P) * invDet;
			if (baryCoords.y < 0 || baryCoords.y > 1)
			{
				return false;
			}

			Q = cross(T, e1);

			baryCoords.z = dot(rd, Q) * invDet;
			baryCoords.x = 1 - (baryCoords.y + baryCoords.z);

			if (baryCoords.z < 0 || baryCoords.x < 0)
			{
				return false;
			}

			t = dot(e2, Q) * invDet;
			if (t < eps)
			{
				return false;
			}

			return true;
		}

		GLM_FUNC_QUALIFIER vec3 translate(const mat4 &m, const vec3 &v)
		{
			vec4 hom = m * vec4(v, 1.0f);
			return vec3(hom) / hom.w;
		}

		GLM_FUNC_QUALIFIER vec3 translaten(const mat4 &m, const vec3 &v)
		{
			return normalize(translate(m, v));
		}
	}
}

#endif //_MATH_H_
