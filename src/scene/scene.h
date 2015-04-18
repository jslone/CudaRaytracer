#ifndef _SCENE_H_
#define _SCENE_H_

#include "assimp/scene.h"
#include "geometry/geometry.h"
#include "math/math.h"

namespace acr {
  
  class Object : Shape {
    public:
      virtual bool intersect(const Ray &r, HitInfo &info);

    	int meshIndex;
    	int numChildren;
    	Object* parent;
      Object** children;
    	math::mat4 globalTransform;
    	math::mat4 localTransform;
      math::mat4 globalNormalTransform;

    	math::mat4 globalInverseTransform;
      math::mat4 globalInverseNormalTransform;
  };

  class Camera {
    public:
      float aspectRatio;
      float horizontalFOV;
      math::mat4 globalTransform;
  };

  class Scene : Shape {
    public:
      struct Args {
        const char* filePath;
      };

      Scene(const Args &args);
      ~Scene();

      virtual bool intersect(const Ray& r, HitInfo &info);
    private:
      void loadScene(const aiScene* scene);
      Object* loadNode(aiNode* node, Object* parent);
      math::mat4& getMathMatrix(aiMatrix4x4 aiMatrix);
      Object* root;
      Camera camera;
  };

} // namespace acr

#endif //_SCENE_H_
