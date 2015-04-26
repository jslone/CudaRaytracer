#ifndef _CAMERA_H_
#define _CAMERA_H_

#include "assimp/scene.h"
#include "math/math.h"

namespace acr
{
	inline math::vec3 getVec3(aiVector3D aivec)
	{
		return math::vec3(aivec.x, aivec.y, aivec.z);
	}

	class Camera
	{
	public:
		Camera() = default;
		Camera(const aiCamera *camera);
		float aspectRatio;
		float horizontalFOV;
		math::vec3 position,forward,up;
	};

	Camera::Camera(const aiCamera *cam)
	{
		aspectRatio = cam->mAspect;
		horizontalFOV = cam->mHorizontalFOV;
		position = getVec3(cam->mPosition);
		up = getVec3(cam->mUp);
		forward = math::normalize(getVec3(cam->mLookAt) - position);
	}

} //namespace acr

#endif //_CAMERA_H_
