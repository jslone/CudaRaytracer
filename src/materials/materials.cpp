#include "materials.h"

namespace acr
{
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
