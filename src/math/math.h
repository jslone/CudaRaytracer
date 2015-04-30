#ifndef _MATH_H_
#define _MATH_H_

#include "glm.hpp"
#include "gtc/matrix_transform.hpp"

namespace acr
{
	namespace math
	{
		using namespace glm;
		
		template<typename genType>
		GLM_FUNC_QUALIFIER genType epsilon()
		{
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
	}
}

#endif //_MATH_H_
