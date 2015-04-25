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

	Material::Material(const aiMaterial *material)
	{
		aiColor3D aiDiffuse,aiAmbient,aiSpecular;
		material->Get(AI_MATKEY_COLOR_DIFFUSE, aiDiffuse);
		material->Get(AI_MATKEY_COLOR_AMBIENT, aiAmbient);
		material->Get(AI_MATKEY_COLOR_SPECULAR, aiSpecular);

		diffuse = getColor3(aiDiffuse);
		ambient = getColor3(aiAmbient);
		specular = getColor3(aiSpecular);

		material->Get(AI_MATKEY_REFRACTI, refractiveIndex);
	}

} // namespace acr

#endif //_MATERIALS_H_
