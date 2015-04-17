#ifndef _SCENE_H_
#define _SCENE_H_

#include "assimp/scene.h"
#include "geometry/geometry.h"
#include "math/math.h"

namespace acr {
  
  class Object {
  	Mesh* mesh;
  	Object** children;
  	int numChildren;
  	Object* parent;
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
  };

} // namespace acr

#endif //_SCENE_H_
