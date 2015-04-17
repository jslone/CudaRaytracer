#ifndef _SCENE_H_
#define _SCENE_H_

#include "assimp/scene.h"
#include "geometry/geometry.h"
#include "math/math.h"

namespace acr {
  
  class Object {
    public:
    	int meshIndex;
    	int numChildren;
    	Object* parent;
      Object** children;
    	math::mat4 globalTransform;
    	math::mat4 localTransform;
    	math::mat4 globalInverseTransform;
  };

  class Scene {
    public:
      struct Args {
        const char* filePath;
      };

      Scene(const Args &args);
      ~Scene();      
    private:
      void loadScene(const aiScene* scene);
      Object* loadNode(aiNode* node, Object* parent);
      math::mat4& getMathMatrix(aiMatrix4x4 aiMatrix);
      Object* root;
  };

} // namespace acr

#endif //_SCENE_H_
