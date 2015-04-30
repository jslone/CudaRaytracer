#ifndef _MATERIALS_H_
#define _MATERIALS_H_

#include "assimp/material.h"
#include "math/math.h"

namespace acr
{
	typedef math::vec3 Color3;
	typedef math::vec4 Color4;

	class Material
	{
		public:
			Material(const aiMaterial *material);
			Color3 ambient,diffuse,specular;
			float refractiveIndex;
	};

	inline Color3 getColor3(aiColor3D aicol)
	{
		return Color3(aicol.r, aicol.g, aicol.b);
	}

} // namespace acr

#endif //_MATERIALS_H_
