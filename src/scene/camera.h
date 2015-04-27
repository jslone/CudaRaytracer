#ifndef _CAMERA_H_
#define _CAMERA_H_

#include "assimp/scene.h"
#include "math/math.h"

namespace acr
{
	class Camera
	{
	public:
		Camera() = default;
		Camera(const aiCamera *camera);
		float aspectRatio;
		float horizontalFOV;
		math::vec3 position,forward,up;
	};
} //namespace acr

#endif //_CAMERA_H_
